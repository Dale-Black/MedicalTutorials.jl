### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 514ab1cc-fb98-11eb-2a1e-c7b264f7066c
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add("Tar")
		Pkg.add("MLDataPattern")
		Pkg.add("Glob")
		Pkg.add("NIfTI")
		Pkg.add("DLPipelines")
		Pkg.add("DataAugmentation")
		Pkg.add("CairoMakie")
		Pkg.add("ImageCore")
		Pkg.add("DataLoaders")
		Pkg.add("Flux")
		Pkg.add("CUDA")
		Pkg.add(url="https://github.com/FluxML/FastAI.jl")
	end
	
	using PlutoUI
	using Tar
	using MLDataPattern
	using Glob
	using NIfTI
	using DLPipelines
	using DataAugmentation
	using DataAugmentation: OneHot, Image
	using CairoMakie
	using ImageCore
	using DataLoaders
	using Flux
	using CUDA
	using FastAI
end

# ╔═╡ 5b2ed761-eb57-44c2-9ed9-bc1e5240af16
TableOfContents()

# ╔═╡ 3cfef02c-00f4-48e9-90b1-4ad43472f397
md"""
## Load data
Part of the [Medical Decathlon Dataset](http://medicaldecathlon.com/)
"""

# ╔═╡ 92776990-345c-4abc-82ad-3cec0bee356f
data_dir = "/Users/daleblack/Google Drive/Datasets/Task02_Heart"

# ╔═╡ bf88491c-a431-4575-a88e-6cc80afce35e
function loadfn_label(p)
	a = NIfTI.niread(string(p)).raw
	convert_a = convert(Array{UInt8}, a)
	return convert_a
end

# ╔═╡ ff620c1f-e08f-4e8d-b538-31a48cea2e0b
function loadfn_image(p)
	a = NIfTI.niread(string(p)).raw
	convert_a = convert(Array{Float32}, a)
	return convert_a
end

# ╔═╡ ae8852a9-5b80-4bda-b58f-3ff3d927bfb4
begin
	
	niftidata_image(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
	niftidata_label(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
	data = (
		niftidata_image(joinpath(data_dir, "imagesTr")),
		niftidata_label(joinpath(data_dir, "labelsTr")),
	)
end

# ╔═╡ 2fd15108-ced5-4dc2-abb4-ceb7f394e718
md"""
## Create `LearningMethod`
"""

# ╔═╡ d2b61982-7829-486e-a0d9-c02fb6d77e74
struct ImageSegmentationSimple <: DLPipelines.LearningMethod
    imagesize
end

# ╔═╡ 69278a55-81b9-4936-9146-f2510774ff22
image_size = (112, 112, 96)

# ╔═╡ 55f8c444-3486-4cd8-a1af-a3774ad55594
method = ImageSegmentationSimple(image_size)

# ╔═╡ c981defe-f337-47ac-8e7b-ea646b4cf482
md"""
### Set up `encode` pipelines
"""

# ╔═╡ 65570158-ee5a-437d-937f-e37f47c07823
begin
	function DLPipelines.encode(
			method::ImageSegmentationSimple,
			context::Training,
			(image, target)::Union{Tuple, NamedTuple}
			)
		
		tfm_proj = RandomResizeCrop(method.imagesize)
		tfm_im = DataAugmentation.compose(
			ImageToTensor(),
			NormalizeIntensity()
			)
		tfm_mask = OneHot()
		
		items = Image(ImageCore.colorview(Gray, image)), MaskMulti(target .+ 1)
		item_im, item_mask = apply(tfm_proj, (items))
		
		return apply(tfm_im, item_im), apply(tfm_mask, item_mask)
	end

	function DLPipelines.encode(
			method::ImageSegmentationSimple,
			context::Validation,
			(image, target)::Union{Tuple, NamedTuple}
			)
		
		tfm_proj = CenterResizeCrop(method.imagesize)
		tfm_im = DataAugmentation.compose(
			ImageToTensor(),
			NormalizeIntensity()
			)
		tfm_mask = OneHot()
		
		items = Image(ImageCore.colorview(Gray, image)), MaskMulti(target .+ 1)
		item_im, item_mask = apply(tfm_proj, (items))
		
		return apply(tfm_im, item_im), apply(tfm_mask, item_mask)
	end
end

# ╔═╡ c85b7bf9-ab6e-45bf-b3e1-a029d715f4d5
md"""
## Create `methoddataloaders`
"""

# ╔═╡ 7ea4e1fa-cbb4-42ee-9c6b-ee801c35f2f2
batchsize = 2

# ╔═╡ 8dc5c540-4545-40d5-9999-2a4d3a853f5c
traindl, validdl = methoddataloaders(data, method, batchsize)

# ╔═╡ f94431fa-89c3-4030-807e-cc61a9977e81
md"""
## Visualize
"""

# ╔═╡ 85de2def-01ee-4fff-bfac-73c8d30ff34d
traindl[1]

# ╔═╡ 92f78dea-2c53-4fe0-aca2-247dbded247a
md"""
## Create  model
"""

# ╔═╡ a458ad0a-b48d-4c31-9f1c-cbe8717cc901
begin
	# 3D layer utilities
	conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=SamePad())
	tran = (stride, in, out) -> ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=SamePad())

	conv1 = (in, out) -> Chain(conv(1, in, out), BatchNorm(out), x -> leakyrelu.(x))
	conv2 = (in, out) -> Chain(conv(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
	tran2 = (in, out) -> Chain(tran(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
end

# ╔═╡ c082b7e4-71d9-4a27-a302-18e543492f3d
begin
	function unet3D(in_chs, lbl_chs)
		# Contracting layers
		l1 = Chain(conv1(in_chs, 4))
		l2 = Chain(l1, conv1(4, 4), conv2(4, 16))
		l3 = Chain(l2, conv1(16, 16), conv2(16, 32))
		l4 = Chain(l3, conv1(32, 32), conv2(32, 64))
		l5 = Chain(l4, conv1(64, 64), conv2(64, 128))

		# Expanding layers
		l6 = Chain(l5, tran2(128, 64), conv1(64, 64))
		l7 = Chain(Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))
		l8 = Chain(Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))
		l9 = Chain(Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))
		l10 = Chain(l9, conv1(4, lbl_chs))
	end
end

# ╔═╡ ca6a29dc-55b8-4238-9ea8-6da9e0372158
md"""
## Create `Learner`
"""

# ╔═╡ 63a851e5-33bd-4756-aa49-d4f872020afc
begin
	model = unet3D(1, 2)
	optimizer = Flux.ADAM(0.01)
	loss_function = Flux.Losses.dice_coeff_loss
end

# ╔═╡ 83c1f0db-0526-407e-a620-3d913061da66
learner = Learner(model, (traindl, validdl), optimizer, loss_function)

# ╔═╡ 103ceb8d-995c-483f-863c-dd4e1f397f3f
md"""
## Train
"""

# ╔═╡ d60a4eee-b796-4f4b-bef8-99df74db4d3a
# fitonecycle!(learner, 1)

# ╔═╡ Cell order:
# ╠═514ab1cc-fb98-11eb-2a1e-c7b264f7066c
# ╠═5b2ed761-eb57-44c2-9ed9-bc1e5240af16
# ╟─3cfef02c-00f4-48e9-90b1-4ad43472f397
# ╠═92776990-345c-4abc-82ad-3cec0bee356f
# ╠═bf88491c-a431-4575-a88e-6cc80afce35e
# ╠═ff620c1f-e08f-4e8d-b538-31a48cea2e0b
# ╠═ae8852a9-5b80-4bda-b58f-3ff3d927bfb4
# ╟─2fd15108-ced5-4dc2-abb4-ceb7f394e718
# ╠═d2b61982-7829-486e-a0d9-c02fb6d77e74
# ╠═69278a55-81b9-4936-9146-f2510774ff22
# ╠═55f8c444-3486-4cd8-a1af-a3774ad55594
# ╟─c981defe-f337-47ac-8e7b-ea646b4cf482
# ╠═65570158-ee5a-437d-937f-e37f47c07823
# ╟─c85b7bf9-ab6e-45bf-b3e1-a029d715f4d5
# ╠═7ea4e1fa-cbb4-42ee-9c6b-ee801c35f2f2
# ╠═8dc5c540-4545-40d5-9999-2a4d3a853f5c
# ╟─f94431fa-89c3-4030-807e-cc61a9977e81
# ╠═85de2def-01ee-4fff-bfac-73c8d30ff34d
# ╟─92f78dea-2c53-4fe0-aca2-247dbded247a
# ╠═a458ad0a-b48d-4c31-9f1c-cbe8717cc901
# ╠═c082b7e4-71d9-4a27-a302-18e543492f3d
# ╟─ca6a29dc-55b8-4238-9ea8-6da9e0372158
# ╠═63a851e5-33bd-4756-aa49-d4f872020afc
# ╠═83c1f0db-0526-407e-a620-3d913061da66
# ╟─103ceb8d-995c-483f-863c-dd4e1f397f3f
# ╠═d60a4eee-b796-4f4b-bef8-99df74db4d3a
