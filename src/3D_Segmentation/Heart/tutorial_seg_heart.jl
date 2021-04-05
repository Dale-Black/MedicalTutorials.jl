### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ eea07192-8f4a-11eb-3996-0d0c407319d8
begin
	let
		# Set up temporary environment
		env = mktempdir()
		import Pkg
		Pkg.activate(env)
		Pkg.Registry.update()
		
		# Download packages
		Pkg.add("PlutoUI")
		Pkg.add("MLDataPattern")
		Pkg.add("Glob")
		Pkg.add("NIfTI")
		Pkg.add("DLPipelines")
		Pkg.add("DataAugmentation")
		Pkg.add("Plots")
		Pkg.add("DataLoaders")
		Pkg.add("Random")
		Pkg.add("Flux")
		Pkg.add("FluxTraining")
		Pkg.add("ImageCore")
	end
	
	# Import packages
	using PlutoUI
	using MLDataPattern
	using Glob
	using NIfTI
	using DLPipelines
	using DataAugmentation
	using DataAugmentation: apply, RandomResizeCrop, CenterResizeCrop, ImageToTensor,
		NormalizeIntensity, OneHot, Image, itemdata, MaskMulti
	using Plots
	using DataLoaders
	using Random
	using Flux
	using ImageCore
end

# ╔═╡ 104240bb-0365-4759-ada3-90de6afb9656
md"""
# Deep Learning With Flux: Heart Segmentation Tutorial
"""

# ╔═╡ 8446a07c-8f4b-11eb-23d6-9b5e1c7d622e
TableOfContents()

# ╔═╡ b4223cbc-8f4a-11eb-2b76-edbdadc14c1b
md"""
## Set up environment, download packages, and import packages
"""

# ╔═╡ 7356a9e4-8f4c-11eb-0dd5-bfcd8c56dee2
md"""
## Set up directory

* Publicly available for download though [Monai](https://docs.monai.io/en/latest/_modules/monai/data/dataset.html) or
* Publicly available for download through the original [Medical Decathlon Segmentation challenge](http://medicaldecathlon.com/)
"""

# ╔═╡ 7e915c2a-8f4c-11eb-1d0d-d382e81949ae
data_dir = "/Users/daleblack/Google Drive/Datasets/Task02_Heart";

# ╔═╡ 49beb3ae-8f4d-11eb-27ce-2765868fd1bc
struct imagesTr
	files::Vector{String}
end

# ╔═╡ 5327d4d4-8f4d-11eb-2f54-51488bedaeb9
struct labelsTr
	files::Vector{String}
end

# ╔═╡ 82f4d65c-8f4c-11eb-08f7-8b16f7f49135
begin
	MLDataPattern.nobs(ds::imagesTr) = length(ds.files)
	MLDataPattern.getobs(ds::imagesTr, idx::Int) = NIfTI.niread(ds.files[idx]).raw
	MLDataPattern.nobs(ds::labelsTr) = length(ds.files)
	MLDataPattern.getobs(ds::labelsTr, idx::Int) = NIfTI.niread(ds.files[idx]).raw
end;

# ╔═╡ 6ad8c12c-8f4d-11eb-2a1d-2b35048f4a0b
train_images = imagesTr(Glob.glob("*.nii.gz", joinpath(data_dir, "imagesTr")))

# ╔═╡ 72f96208-8f4d-11eb-1a7d-bb1723acbd78
train_labels = labelsTr(Glob.glob("*.nii.gz", joinpath(data_dir, "labelsTr")))

# ╔═╡ 79477078-8f4d-11eb-33f4-b97aa711cad2
train_files, val_files = MLDataPattern.splitobs((train_images, train_labels), 0.8)

# ╔═╡ ec0fb38e-8f4d-11eb-3995-09b8740d02a1
md"""
Double check that the data files are set up properly. The files should contain multiple observations (images) and `MLDataPattern.getobs` should load the one or more files as pure Julia arrays
"""

# ╔═╡ f212b670-8f4d-11eb-3219-4b53531740c2
let
	x, y = MLDataPattern.getobs(train_files, 2)
	typeof(x), size(x), size(y)
end

# ╔═╡ 38dfcad4-8f4e-11eb-06bd-13a47c425053
md"""
## Set up data loading pipelines
* Follow [this workflow](https://www.notion.so/Deep-learning-workflow-Ecosystem-overview-c5648bd1951b404d911fe705eced0e41) with some modifications
"""

# ╔═╡ da0136e4-8f50-11eb-3318-f99425553e1e
abstract type
	ImageSegmentationTask <: DLPipelines.LearningTask
end

# ╔═╡ b51c926c-8f4e-11eb-0a3c-65df9e8b2f0c
struct ImageSegmentationSimple <: DLPipelines.LearningMethod{ImageSegmentationTask}
	imagesize
end

# ╔═╡ 197d2bb2-f72b-461e-a46a-e5ac303cb28d
md"""
### Create `AddChannel()` transform
We are working with 3D grayscale images that need to be fed into a neural net as `size = (x, y, z, 1)`
"""

# ╔═╡ 1070b8f8-2fa4-4612-a520-af71459582a1
struct MapItemData <: Transform
    f
end

# ╔═╡ 19f869d4-f4b5-4aae-959e-eb4f0ef8c401
begin
	DataAugmentation.apply(tfm::MapItemData, item::DataAugmentation.AbstractItem; randstate = nothing) = DataAugmentation.setdata(item, tfm.f(itemdata(item)))
	
	DataAugmentation.apply(tfm::MapItemData, item::DataAugmentation.Image; randstate = nothing) = DataAugmentation.setdata(item, tfm.f(itemdata(item)))
end

# ╔═╡ fd999115-2464-46a1-8e34-f8cfbbc1ee35
AddChannel() = MapItemData(a -> reshape(a, size(a)..., 1))

# ╔═╡ 3dc50780-8f4e-11eb-3f29-c74b44dd1528
begin
	imsize = (112, 112, 96)
	method = ImageSegmentationSimple(imsize)
end;

# ╔═╡ be0aa160-64ed-4499-91a7-f011990862a5
md"""
### Set up `ecodeinput()` pipelines
Both the training set and the validation set require a pipeline. The validation set should not have any Random transforms whereas the training set can contain random transforms
"""

# ╔═╡ 15c2e164-8f54-11eb-1315-6b553b75d0ed
begin
	function DLPipelines.encodeinput(
			method::ImageSegmentationSimple,
			context::Training,
			image)
		tfm = RandomResizeCrop(method.imagesize) |> ImageToTensor() |> NormalizeIntensity() |> AddChannel() 
		return apply(tfm, Image(ImageCore.colorview(Gray, image))) |> itemdata
	end
	
	function DLPipelines.encodeinput(
			method::ImageSegmentationSimple,
			context::Validation,
			image)
		tfm = CenterResizeCrop(method.imagesize) |> ImageToTensor() |> NormalizeIntensity() |> AddChannel() 
		return apply(tfm, Image(ImageCore.colorview(Gray, image))) |> itemdata
	end
end

# ╔═╡ d9a3589a-2c7a-4caf-8e62-54a6dd95dff9
md"""
### Set up `ecodetarget()` pipelines
Both the training set and the validation set require a pipeline. The validation set should not have any Random transforms whereas the training set can contain random transforms
"""

# ╔═╡ 590980a8-90d8-11eb-275b-b3eba753c87e
begin
	function DLPipelines.encodetarget(
			method::ImageSegmentationSimple,
			context::Training,
			image)
		tfm = RandomResizeCrop(method.imagesize) |> OneHot()
		return apply(tfm, MaskMulti(image .+ 1)) |> itemdata
	end
	
	function DLPipelines.encodetarget(
			method::ImageSegmentationSimple,
			context::Validation,
			image)
		tfm = CenterResizeCrop(method.imagesize) |> OneHot()
		return apply(tfm, MaskMulti(image .+ 1)) |> itemdata
	end
end

# ╔═╡ 3443266e-5d97-44b8-96a7-43eefb42c7c6
md"""
### Set up dataset
"""

# ╔═╡ 31c6e194-8f54-11eb-1e50-df59322c9a29
begin
	methoddata_train = DLPipelines.MethodDataset(train_files, method, Training())
	methoddata_valid = DLPipelines.MethodDataset(val_files, method, Validation())
end;

# ╔═╡ 3afa3c22-8f54-11eb-0439-15fcbcfabfc8
md"""
Double check that the data processing works as expected. After applying the transforms, the first images `x, y` should now be of `size = (112, 112, 96, 1)`

*Note `imsize...` is the same as `(112, 112, 96)`*
"""

# ╔═╡ 3e163f28-8f54-11eb-0af0-ddcbab45cb82
let
	x, y = MLDataPattern.getobs(methoddata_train, 1)
	@assert size(x) == (imsize..., 1)
	@assert size(y) == (imsize..., 2)
end

# ╔═╡ acd6a86a-c6ad-4c08-8c3d-dbe0424e7d6a
let
	x, y = MLDataPattern.getobs(methoddata_train, 1)
	size(x)
end

# ╔═╡ 46b192c8-90a9-11eb-1b7d-954c786d4a7e
md"""
## Plot data
"""

# ╔═╡ 4c67e90e-90a9-11eb-26df-cb5b22a1991d
x, y = MLDataPattern.getobs(methoddata_valid, 2);

# ╔═╡ 60a80e7e-90a9-11eb-1bbb-41ef1c27ef25
md"""
$(@bind a Slider(1:96, default=62, show_value=true))
"""

# ╔═╡ 716f8acc-90a9-11eb-2889-53a02559cd8a
Plots.heatmap(x[:, :, a, 1], c = :grays)

# ╔═╡ 7f8f83d2-90a9-11eb-30f7-7d71a11e1bc7
Plots.heatmap(y[:, :, a, 2], c = :grays)

# ╔═╡ 02b9268a-90ac-11eb-1d12-915cce1e75c4
md"""
## Load data into a DataLoader
This allows us to iterate over batches of data easily
"""

# ╔═╡ 0a77ead2-90ac-11eb-3d26-4fe3dfab13fc
begin
	train_loader = DataLoaders.DataLoader(methoddata_train, 4)
	val_loader = DataLoaders.DataLoader(methoddata_valid, 2)
end;

# ╔═╡ 2a420dac-90e8-11eb-14db-6ffe83fd1e32
md"""
Double check the size of the batches
"""

# ╔═╡ 4e5d8958-90e8-11eb-032d-e98315406db7
for (xs, ys) in train_loader
	@assert size(xs) == (imsize..., 1, 4)
	@assert size(ys) == (imsize..., 2, 4)
end

# ╔═╡ 32ffc6fa-90e8-11eb-01bc-1ba1e0c8ca22
for (xs, ys) in val_loader
	@assert size(xs) == (imsize..., 1, 2)
	@assert size(ys) == (imsize..., 2, 2)
end

# ╔═╡ 7dc9da92-90a8-11eb-1ca2-296e3195a785
md"""
## Set deterministic training for reproducibility
"""

# ╔═╡ 7d9f5986-90a8-11eb-191e-79fde3e68583
Random.seed!(1);

# ╔═╡ ab2f02a2-90a8-11eb-375e-afb896cd9699
md"""
## Create model
* Create a [Unet model](https://arxiv.org/abs/1505.04597)
* Adapted from [link](https://github.com/DhairyaLGandhi/UNet.jl)
* Complete set of [medical specific deep learning models](https://github.com/Dale-Black/MedicalModels.jl)
*Note: this a 3-Dimensional UNet with residual `Parallel(+, ...)` connections*
"""

# ╔═╡ adc61bae-90a8-11eb-3558-ed3a428a38b8
begin
	# 3D layer utilities
	conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=SamePad())
	tran = (stride, in, out) -> ConvTranspose((3, 3, 3), in=>out, stride=stride, pad=SamePad())

	conv1 = (in, out) -> Chain(conv(1, in, out), BatchNorm(out), x -> leakyrelu.(x))
	conv2 = (in, out) -> Chain(conv(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
	tran2 = (in, out) -> Chain(tran(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
end;

# ╔═╡ 0d73fbf2-90a9-11eb-1c2c-f7bad2819c63
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

# ╔═╡ a6ee1864-90ac-11eb-1730-a74e8861d57d
md"""
## Train model

Here we will use the basics for training
* Dice loss
* ADAM optimizer
* etc
"""

# ╔═╡ 311fd450-90ad-11eb-0f74-ad86115baf94
begin
	model = unet3D(1, 2)
	ps = Flux.params(model)
	loss_function = Flux.Losses.dice_coeff_loss
	optimizer = Flux.ADAM(0.01)
end;

# ╔═╡ 9d416b26-90ad-11eb-2c5f-f56340e14275
# begin
# 	max_epochs = 1
# 	epoch_loss_values = []
# 	metric_values = []

# 	for epoch in 1:max_epochs
# 		epoch_loss = 0
# 		steps = 0
		
# 		# Training loop for training data
# 		for (xs, ys) in train_loader
# 			steps += 1
# 			gs = Flux.gradient(ps) do
# 				ŷs = model(xs)
# 				loss = loss_function(ŷs[:, :, :, 2, :], ys[:, :, :, 2, :])
# 				epoch_loss += loss
# 				return loss
# 			end
# 			Flux.update!(optimizer, ps, gs)
# 		end
# 		epoch_loss = (epoch_loss / steps)
# 		push!(epoch_loss_values, epoch_loss)
# 	end
# end

# ╔═╡ bb4bc040-90ed-11eb-3893-1ff7940dcdf7
epoch_loss_values

# ╔═╡ Cell order:
# ╟─104240bb-0365-4759-ada3-90de6afb9656
# ╠═8446a07c-8f4b-11eb-23d6-9b5e1c7d622e
# ╟─b4223cbc-8f4a-11eb-2b76-edbdadc14c1b
# ╠═eea07192-8f4a-11eb-3996-0d0c407319d8
# ╟─7356a9e4-8f4c-11eb-0dd5-bfcd8c56dee2
# ╠═7e915c2a-8f4c-11eb-1d0d-d382e81949ae
# ╠═49beb3ae-8f4d-11eb-27ce-2765868fd1bc
# ╠═5327d4d4-8f4d-11eb-2f54-51488bedaeb9
# ╠═82f4d65c-8f4c-11eb-08f7-8b16f7f49135
# ╠═6ad8c12c-8f4d-11eb-2a1d-2b35048f4a0b
# ╠═72f96208-8f4d-11eb-1a7d-bb1723acbd78
# ╠═79477078-8f4d-11eb-33f4-b97aa711cad2
# ╟─ec0fb38e-8f4d-11eb-3995-09b8740d02a1
# ╠═f212b670-8f4d-11eb-3219-4b53531740c2
# ╟─38dfcad4-8f4e-11eb-06bd-13a47c425053
# ╠═da0136e4-8f50-11eb-3318-f99425553e1e
# ╠═b51c926c-8f4e-11eb-0a3c-65df9e8b2f0c
# ╟─197d2bb2-f72b-461e-a46a-e5ac303cb28d
# ╠═1070b8f8-2fa4-4612-a520-af71459582a1
# ╠═19f869d4-f4b5-4aae-959e-eb4f0ef8c401
# ╠═fd999115-2464-46a1-8e34-f8cfbbc1ee35
# ╠═3dc50780-8f4e-11eb-3f29-c74b44dd1528
# ╟─be0aa160-64ed-4499-91a7-f011990862a5
# ╠═15c2e164-8f54-11eb-1315-6b553b75d0ed
# ╟─d9a3589a-2c7a-4caf-8e62-54a6dd95dff9
# ╠═590980a8-90d8-11eb-275b-b3eba753c87e
# ╟─3443266e-5d97-44b8-96a7-43eefb42c7c6
# ╠═31c6e194-8f54-11eb-1e50-df59322c9a29
# ╟─3afa3c22-8f54-11eb-0439-15fcbcfabfc8
# ╠═3e163f28-8f54-11eb-0af0-ddcbab45cb82
# ╠═acd6a86a-c6ad-4c08-8c3d-dbe0424e7d6a
# ╟─46b192c8-90a9-11eb-1b7d-954c786d4a7e
# ╠═4c67e90e-90a9-11eb-26df-cb5b22a1991d
# ╟─60a80e7e-90a9-11eb-1bbb-41ef1c27ef25
# ╠═716f8acc-90a9-11eb-2889-53a02559cd8a
# ╠═7f8f83d2-90a9-11eb-30f7-7d71a11e1bc7
# ╟─02b9268a-90ac-11eb-1d12-915cce1e75c4
# ╠═0a77ead2-90ac-11eb-3d26-4fe3dfab13fc
# ╟─2a420dac-90e8-11eb-14db-6ffe83fd1e32
# ╠═4e5d8958-90e8-11eb-032d-e98315406db7
# ╠═32ffc6fa-90e8-11eb-01bc-1ba1e0c8ca22
# ╟─7dc9da92-90a8-11eb-1ca2-296e3195a785
# ╠═7d9f5986-90a8-11eb-191e-79fde3e68583
# ╟─ab2f02a2-90a8-11eb-375e-afb896cd9699
# ╠═adc61bae-90a8-11eb-3558-ed3a428a38b8
# ╠═0d73fbf2-90a9-11eb-1c2c-f7bad2819c63
# ╟─a6ee1864-90ac-11eb-1730-a74e8861d57d
# ╠═311fd450-90ad-11eb-0f74-ad86115baf94
# ╠═9d416b26-90ad-11eb-2c5f-f56340e14275
# ╠═bb4bc040-90ed-11eb-3893-1ff7940dcdf7
