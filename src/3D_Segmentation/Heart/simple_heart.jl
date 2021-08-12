### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ fe5bad56-fb95-11eb-3d20-91c2f7fd23f1
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("PlutoUI")
		Pkg.add("Tar")
		Pkg.add("Glob")
		Pkg.add("NIfTI")
		Pkg.add("CairoMakie")
		Pkg.add("ImageCore")
		Pkg.add("DataLoaders")
		Pkg.add("Flux")
		Pkg.add(url="https://github.com/FluxML/FastAI.jl")
	end
	
	using PlutoUI
	using Tar
	using Glob
	using NIfTI
	using CairoMakie
	using ImageCore
	using DataLoaders
	using Flux
	using FastAI
end

# ╔═╡ 2e13d1cf-fd2c-444a-89b7-46f0bf7243f6
TableOfContents()

# ╔═╡ 7b902334-cf1b-43fe-a802-de6922c5d517
md"""
## Load data
"""

# ╔═╡ 3ede3af1-7ed2-4dc2-89d3-a5a19770ee2c
data_dir = "/Users/daleblack/Google Drive/Datasets/Task02_Heart"

# ╔═╡ 4a22f0ae-2107-4f45-8de7-bbf170e9ec8b
function loadfn_label(p)
	a = NIfTI.niread(string(p)).raw
	convert_a = convert(Array{UInt8}, a)
	return convert_a
end

# ╔═╡ 1a3599d3-2b2e-4a13-8f82-403894bba395
function loadfn_image(p)
	a = NIfTI.niread(string(p)).raw
	convert_a = convert(Array{Float32}, a)
	return convert_a
end

# ╔═╡ 58ad98fc-768a-469e-baf3-decf7cd9fd8f
begin
	niftidata_image(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
	niftidata_label(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
	data = (
		niftidata_image(joinpath(data_dir, "imagesTr")),
		niftidata_label(joinpath(data_dir, "labelsTr")),
	)
end

# ╔═╡ 2313ccd1-3aa5-4c6f-926f-c86579b58cf0
md"""
## Create `BlockMethod`
"""

# ╔═╡ 540d5e85-e240-4bb4-b638-282a87a2a64f
classes = [0, 1]

# ╔═╡ c18af89c-87d6-44ac-abd1-02d51478b0b8
method = BlockMethod(
    (FastAI.Image{3}(), FastAI.Mask{3}(classes),
    (ProjectiveTransforms((64, 64, 64)), ImagePreprocessing(), OneHot()),
))

# ╔═╡ f273e655-aeee-448a-ac02-a5d49ed9a34d
md"""
## Create `methoddataloaders`
"""

# ╔═╡ 24b7bb14-69a4-4c1c-af80-4a328af1296c
batchsize = 2

# ╔═╡ 02a25708-8513-4f26-a6a5-d8fd2e35853f
traindl, validdl = methoddataloaders(data, method, batchsize)

# ╔═╡ 0bd0472f-f053-4f3c-b7d3-725e0b868754
md"""
## Create model
"""

# ╔═╡ 74edfe3e-1ed3-48ef-ac00-29b38b7b065c
begin
	# 3D layer utilities
	conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=SamePad())
	tran = (stride, in, out) -> ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=SamePad())

	conv1 = (in, out) -> Chain(conv(1, in, out), BatchNorm(out), x -> leakyrelu.(x))
	conv2 = (in, out) -> Chain(conv(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
	tran2 = (in, out) -> Chain(tran(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
end

# ╔═╡ 674f7aa7-5d4b-4e70-96b6-7ec01398b2e5
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

# ╔═╡ 8f5a3814-916e-415a-9a93-b5421a0eeb9c
md"""
## Create `Learner`
"""

# ╔═╡ b84511ee-eec6-4864-8e21-7aeb8ec44da8
begin
	model = unet3D(1, 2)
	optimizer = Flux.ADAM(0.01)
	loss_function = Flux.Losses.dice_coeff_loss
end

# ╔═╡ 79548481-fd0f-4276-910a-52ac2ea5b8e0
learner = Learner(model, (traindl, validdl), optimizer, loss_function)

# ╔═╡ bbb9a2dc-c583-442b-86c9-8c1168a92b7c
md"""
## Train
"""

# ╔═╡ ded6ecc8-bba4-4507-b95b-a3ae37974415


# ╔═╡ Cell order:
# ╠═fe5bad56-fb95-11eb-3d20-91c2f7fd23f1
# ╠═2e13d1cf-fd2c-444a-89b7-46f0bf7243f6
# ╟─7b902334-cf1b-43fe-a802-de6922c5d517
# ╠═3ede3af1-7ed2-4dc2-89d3-a5a19770ee2c
# ╠═4a22f0ae-2107-4f45-8de7-bbf170e9ec8b
# ╠═1a3599d3-2b2e-4a13-8f82-403894bba395
# ╠═58ad98fc-768a-469e-baf3-decf7cd9fd8f
# ╟─2313ccd1-3aa5-4c6f-926f-c86579b58cf0
# ╠═540d5e85-e240-4bb4-b638-282a87a2a64f
# ╠═c18af89c-87d6-44ac-abd1-02d51478b0b8
# ╟─f273e655-aeee-448a-ac02-a5d49ed9a34d
# ╠═24b7bb14-69a4-4c1c-af80-4a328af1296c
# ╠═02a25708-8513-4f26-a6a5-d8fd2e35853f
# ╟─0bd0472f-f053-4f3c-b7d3-725e0b868754
# ╠═74edfe3e-1ed3-48ef-ac00-29b38b7b065c
# ╠═674f7aa7-5d4b-4e70-96b6-7ec01398b2e5
# ╟─8f5a3814-916e-415a-9a93-b5421a0eeb9c
# ╠═b84511ee-eec6-4864-8e21-7aeb8ec44da8
# ╠═79548481-fd0f-4276-910a-52ac2ea5b8e0
# ╟─bbb9a2dc-c583-442b-86c9-8c1168a92b7c
# ╠═ded6ecc8-bba4-4507-b95b-a3ae37974415
