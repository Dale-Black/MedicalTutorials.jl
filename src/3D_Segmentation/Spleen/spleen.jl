### A Pluto.jl notebook ###
# v0.12.18

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

# ╔═╡ cd611ea8-4ba4-11eb-025f-2db195ae152b
begin
	
	
	using Pkg
	Pkg.activate(".")
	
	### Uncomment the lines below to install packages. Only neccessary once
	
	# ] add NIfTI
	# ] add Glob
	# ] add MLDataPattern
	# ] add https://github.com/lorenzoh/DLPipelines.jl
	# ] add https://github.com/lorenzoh/DataAugmentation.jl.git
	# ] add PlutoUI
	# ] add Plots
	# ] add Flux
	# ] add Functors
	
	
	using NIfTI
	using Glob
	using MLDataPattern
	using DLPipelines
	using DataAugmentation
	using Random: seed!
	using PlutoUI
	using Plots
	using Flux
	using Functors
	
	
end

# ╔═╡ 27add6de-3922-11eb-0955-156a939a344f
md"""
## Spleen 3D Segmentation with Julia and FluxTraining.jl
This tutorial shows how to use Julia and FluxTraining.jl to train a model to segment the spleen in a publicly available CT dataset. Adapted from [link](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb)
"""

# ╔═╡ 528e6bb6-3922-11eb-1961-41fc9fb4a094
md"""
## Set up environment
"""

# ╔═╡ efe8a1fc-5b84-11eb-27bf-23d304283029
html"""<style>
main {
	max-width: 1500px;
}
"""

# ╔═╡ 5e89f012-4ba5-11eb-3e35-63a1d62e3e94
pwd()

# ╔═╡ 96121a84-3924-11eb-1cbb-0de4f3c73b0c
md"""
## Set up data directory
* Publicly available for download though Monai or
* Publicly available for download through the original Medical Decathlon Segmentation challenge
"""

# ╔═╡ ef664078-3925-11eb-3b3d-438868228269
data_dir = "/Users/daleblack/Google Drive/Datasets/spleen/Task09_Spleen";

# ╔═╡ 3ebe6c74-3926-11eb-2a01-f951f7a6849c
begin 
	
	
	struct imagesTr
		files::Vector{String}
	end
	struct labelsTr
		files::Vector{String}
	end
	
	
end

# ╔═╡ 4a8101aa-3926-11eb-1d4b-0721530a9423
begin

	
	MLDataPattern.nobs(ds::imagesTr) = length(ds.files)
	MLDataPattern.getobs(ds::imagesTr, idx::Int) = (niread(ds.files[idx]).raw)


	MLDataPattern.nobs(ds::labelsTr) = length(ds.files)
	MLDataPattern.getobs(ds::labelsTr, idx::Int) = (niread(ds.files[idx]).raw)
	
	
end

# ╔═╡ 57fd3a7e-3926-11eb-0e3f-858f616cea60
begin

	
	train_images = imagesTr(glob("*.nii.gz", joinpath(data_dir, "imagesTr")))
	train_labels = labelsTr(glob("*.nii.gz", joinpath(data_dir, "labelsTr")))
	train_files, val_files = splitobs((train_images, train_labels), 0.8)


end;

# ╔═╡ 0ea7faf0-4acc-11eb-2ffd-7bf66092c013
md"""
Double check that the data files are set up properly. The files should contain multiple observations (images) and `getobs` should load the files as pure Julia arrays
"""

# ╔═╡ cd647910-4acb-11eb-2ee7-972c711247dc
begin
	
	let
		x, y = getobs(train_files, 2)
		typeof(x), size(x), size(y)
	end
	
	
end

# ╔═╡ 99e83cd8-3929-11eb-0a0e-e1d5780c77b4
md"""
## Set up data processing
* Follow [this workflow](https://www.notion.so/Deep-learning-workflow-Ecosystem-overview-c5648bd1951b404d911fe705eced0e41)
* TODO
  * Add `Orientation` [transform](https://docs.monai.io/en/latest/transforms.html?highlight=orientation#monai.transforms.Orientation)
  * Add `CropForeground` [transform](https://docs.monai.io/en/latest/transforms.html#cropforeground)
"""

# ╔═╡ 43d816ce-4a40-11eb-1647-f37930d8eb8a
begin
	
	
	abstract type ImageSegmentationTask <: DLPipelines.Task end
	
	struct ImageSegmentationSimple <: DLPipelines.Method{ImageSegmentationTask}
		imagesize
	end
	
	method = ImageSegmentationSimple((512, 512, 96))
	
	
end;

# ╔═╡ 797e08be-4a41-11eb-0c4e-1df9d65735b6
begin
	
	
	
	function DLPipelines.encodeinput(
			method::ImageSegmentationSimple,
			context::Training,
			image)
		tfm = RandomResizeCrop(method.imagesize) |> NormalizeIntensity() |> AddChannel()
		return apply(tfm, Image(image)) |> itemdata
	end
	
	function DLPipelines.encodeinput(
			method::ImageSegmentationSimple,
			context::Validation,
			image)
		tfm = CenterResizeCrop(method.imagesize) |> NormalizeIntensity() |> AddChannel()
		return apply(tfm, Image(image)) |> itemdata
	end
	

end

# ╔═╡ ccbd3738-53dc-11eb-27cb-4daae45f3216
begin
	
	
	function DLPipelines.encodetarget(
			method::ImageSegmentationSimple,
			context::Training,
			image)
		tfm = RandomResizeCrop(method.imagesize) |> AddChannel()
		return apply(tfm, MaskBinary(reinterpret(Bool, image))) |> itemdata
	end
	
	function DLPipelines.encodetarget(
			method::ImageSegmentationSimple,
			context::Validation,
			image)
		tfm = CenterResizeCrop(method.imagesize) |> AddChannel()
		return apply(tfm, MaskBinary(reinterpret(Bool, image))) |> itemdata
	end
	

end

# ╔═╡ 794f97cc-4a41-11eb-08f6-e1487743247d
begin
	
	
	methoddata_train = DLPipelines.MethodDataset(train_files, method, Training())
	methoddata_valid = DLPipelines.MethodDataset(val_files, method, Validation())


end

# ╔═╡ 893b0544-4ace-11eb-37d7-6da2eb2ec12d
md"""
Double check that the data processing works as expected. After applying the transforms, the first images `(x, y)` should go from size == `(512, 512, 55)` to size = `(512, 512, 96, 1)`
"""

# ╔═╡ 3c77c182-4acb-11eb-22c5-9f100c192b0d
begin

	let
		x, y = getobs(methoddata_train, 1)
		size(x) == (512, 512, 96, 1)
		size(y) == (512, 512, 96, 1)
	end
	
	
end

# ╔═╡ 96aef858-53dd-11eb-2b07-d17a14c76ed5
md"""
## Plot data
"""

# ╔═╡ 14ca4132-53e5-11eb-2027-75733a1fac04
x, y = getobs(methoddata_valid, 8);

# ╔═╡ 552d6304-53e4-11eb-125b-c18befd4ba5e
md"""
$(@bind a Slider(1:96))
"""

# ╔═╡ 9eaf316c-53dd-11eb-0d26-9f045503a6b6
heatmap(x[:, :, a, 1], c = :grays)

# ╔═╡ 029d10ca-53e5-11eb-329b-f73a2fbce197
heatmap(y[:, :, a, 1], c = :grays)

# ╔═╡ 62148508-3926-11eb-34d2-a7bedab25f30
md"""
## Set deterministic training for reproducibility
"""

# ╔═╡ c01c2eb6-4acf-11eb-2a85-2fe31d094684
seed!(1);

# ╔═╡ 9ea1d47e-53e5-11eb-0ab5-117b5c7a8422
md"""
## Create model
* Create a [Unet model](https://arxiv.org/abs/1505.04597)
* Adapted from [link](https://github.com/DhairyaLGandhi/UNet.jl)

*Note: this a 3-Dimensional UNet*
"""

# ╔═╡ c59a49de-5889-11eb-14a7-afabd132719b
expand_dims(x, n::Int) = reshape(x, ones(Int64,n)..., size(x)...)

# ╔═╡ f653f67e-588c-11eb-01d4-5f3c896ed5b2
with_terminal() do
	x = rand(Float32, 20, 20, 20, 1, 1)
	x_expand = expand_dims(x, 2)
	println("x size: ", size(x))
	println("x after expand: ", size(x_expand))
end

# ╔═╡ bc5228ce-5891-11eb-29eb-e9bbf5ed0510
function squeeze(x)
	if size(x)[end] == 1 && size(x)[end-1] != 1
		# For the case BATCH_SIZE = 1 and Channels != 1
		int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)..., 1)
	elseif size(x)[end] != 1 && size(x)[end-1] == 1
		# For the case BATCH_SIZE != 1 and Channels = 1
		int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)..., 1, :)
	elseif size(x)[end] == 1 && size(x)[end-1] == 1
		# For the case BATCH_SIZE = 1 and Channels = 1
		int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)..., 1, 1)
	else
		size(x)[end] != 1 && size(x)[end-1] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    end
end

# ╔═╡ 480897d2-588c-11eb-2894-699a81fc1358
with_terminal() do
	x = rand(Float32, 1, 1, 1, 20, 20, 20)
	x_squeeze = squeeze(x)
	println("x size: ", size(x))
	println("x after squeeze: ", size(x_squeeze))
	
	y = rand(Float32, 20, 20, 20, 1, 1)
	y_squeeze = squeeze(y)
	println()
	println("y size: ", size(y))
	println("y after squeeze: ", size(y_squeeze))
	
	z = rand(Float32, 1, 20, 20, 20, 1)
	z_squeeze = squeeze(z)
	println()
	println("\nz size: ", size(z))
	println("z after squeeze: ", size(z_squeeze))
end

# ╔═╡ 9ea0f536-579b-11eb-1709-95271e9a775e
function BatchNormWrap(out_ch)
    Chain(x -> expand_dims(x, 3), BatchNorm(out_ch), x -> squeeze(x))
end

# ╔═╡ 32e76378-588d-11eb-0cfe-79706afc57f4
with_terminal() do
	x = rand(Float32, 20, 20, 20, 1, 1)
	x_BNW = BatchNormWrap(1)(x)
	println("x size: ", size(x))
	println("x size after BatchNormWrap: ", size(x_BNW)) # Doesn't change dimensions
end

# ╔═╡ 1ccd9a18-5889-11eb-3a76-b10e73357829
UNetConvBlock(in_chs, out_chs, kernel = (3, 3, 3)) = 
	Chain(
		Conv(kernel, in_chs => out_chs, pad = (1, 1, 1)), 
		BatchNormWrap(out_chs),
		x -> leakyrelu.(x, 0.2f0))

# ╔═╡ b5b7abd2-588d-11eb-17f0-8dd8860d441a
with_terminal() do
	x = rand(Float32, 20, 20, 20, 1, 1)
	x_UNCB = UNetConvBlock(1, 16)(x)
	println("x size: ", size(x))
	println("x size after UNetConvBlock: ", size(x_UNCB)) # Changes channel dimensions
end

# ╔═╡ e18b2054-588d-11eb-30a3-3b339c1370b3
ConvDown(in_chs, out_chs, kernel = (4, 4, 4)) = Chain(
    Conv(kernel, in_chs => out_chs, pad = (1, 1, 1), stride = (2, 2, 2)),
    BatchNormWrap(out_chs),
    x -> leakyrelu.(x, 0.2f0))

# ╔═╡ e8d40100-588d-11eb-16dc-77fd936d9f44
with_terminal() do
	x = rand(Float32, 20, 20, 20, 1, 1)
	x_CD = ConvDown(1, 16)(x)
	println("x size: ", size(x))
	println("x size after ConvDown: ", size(x_CD)) # Halves W, H, D dimensions
end

# ╔═╡ 3474dc06-5889-11eb-28a6-b19e3b392af8
begin
	
	
	struct UNetUpBlock
    	upsample
	end

	@functor UNetUpBlock
	
	UNetUpBlock(in_chs::Int, out_chs::Int; kernel = (3, 3, 3), p = 0.5f0) = 
		UNetUpBlock(
			Chain(
				x -> leakyrelu.(x, 0.2f0),
				ConvTranspose((2, 2, 2), in_chs => out_chs, stride = (2, 2, 2)),
				BatchNormWrap(out_chs),
				Dropout(p)))
	
	function (u::UNetUpBlock)(x, bridge)
		x = u.upsample(x)
		return cat(x, bridge, dims = 4)
	end
	
	struct Unet
	  conv_down_blocks
	  conv_blocks
	  up_blocks
	end
	
	@functor Unet
	
	function Unet(channels::Int = 1, labels::Int = channels)
		conv_down_blocks = Chain(
			ConvDown(64, 64),
			ConvDown(128, 128),
			ConvDown(256, 256),
			ConvDown(512, 512))

		conv_blocks = Chain(
			UNetConvBlock(channels, 3),
			UNetConvBlock(3, 64),
			UNetConvBlock(64, 128),
			UNetConvBlock(128, 256),
			UNetConvBlock(256, 512),
			UNetConvBlock(512, 1024),
			UNetConvBlock(1024, 1024))

		up_blocks = Chain(
			UNetUpBlock(1024, 512),
			UNetUpBlock(1024, 256),
			UNetUpBlock(512, 128),
			UNetUpBlock(256, 64,p = 0.0f0),
			Chain(x -> leakyrelu.(x, 0.2f0),
			Conv((1, 1, 1), 128 => labels)))

		Unet(conv_down_blocks, conv_blocks, up_blocks)
	  end
	
	function (u::Unet)(x::AbstractArray)
		op = u.conv_blocks[1:2](x)

		x1 = u.conv_blocks[3](u.conv_down_blocks[1](op))
		x2 = u.conv_blocks[4](u.conv_down_blocks[2](x1))
		x3 = u.conv_blocks[5](u.conv_down_blocks[3](x2))
		x4 = u.conv_blocks[6](u.conv_down_blocks[4](x3))

		up_x4 = u.conv_blocks[7](x4)

		up_x1 = u.up_blocks[1](up_x4, x3)
		up_x2 = u.up_blocks[2](up_x1, x2)
		up_x3 = u.up_blocks[3](up_x2, x1)
		up_x5 = u.up_blocks[4](up_x3, op)
		tanh.(u.up_blocks[end](up_x5))
	end
	
end

# ╔═╡ 1a5d9a88-588e-11eb-2b0c-3115e94329db
with_terminal() do
	x = rand(Float32, 10, 10, 10, 16, 1)
	y = rand(Float32, 20, 20, 20, 16, 1)
	println("x size: ", size(x))
	println("y size: ", size(y))
	
	println()
	block = UNetUpBlock(16, 16)
	new = (block)(x, y)
	println("size after UNetUpBlock: ", size(new))
end

# ╔═╡ 4f019f5e-5894-11eb-0810-d343b6345fc6
with_terminal() do
	x = rand(Float32, 64, 64, 64, 1, 1)
	u = Unet(1, 1)
	println("x size: ", size(x))
	
	println()
	op = u.conv_blocks[1:2](x)
	println("op size: ", size(op))
	
	println()
	x1 = u.conv_blocks[3](u.conv_down_blocks[1](op))
	x2 = u.conv_blocks[4](u.conv_down_blocks[2](x1))
	x3 = u.conv_blocks[5](u.conv_down_blocks[3](x2))
	x4 = u.conv_blocks[6](u.conv_down_blocks[4](x3))
	println("x1 size: ", size(x1))
	println("x2 size: ", size(x2))
	println("x3 size: ", size(x3))
	println("x4 size: ", size(x4))
	
	println()
	up_x4 = u.conv_blocks[7](x4)
	println("up_x4 size: ", size(up_x4))
	
	println()
	up_x1 = u.up_blocks[1](up_x4, x3)
	up_x2 = u.up_blocks[2](up_x1, x2)
	up_x3 = u.up_blocks[3](up_x2, x1)
	up_x5 = u.up_blocks[4](up_x3, op)
	println("up_x1 size: ", size(up_x1))
	println("up_x2 size: ", size(up_x2))
	println("up_x3 size: ", size(up_x3))
	println("up_x5 size: ", size(up_x5))
	
	println()
	last = tanh.(u.up_blocks[end](up_x5))
	println("last size: ", size(last))
end

# ╔═╡ d4ff40a4-5b85-11eb-1216-ed4c6029426c
md"""
Load an entire 5D array `(width, height, depth, channels, batch_size)` into the model all at one. The array size should be unchanged but the values should be different
"""

# ╔═╡ 5acdb270-5b85-11eb-3590-afaf2ac61d88
with_terminal() do
	model = Unet(1, 1)
	x = rand(Float32, 64, 64, 64, 1, 1)
	x̂ = model(x)
	println("size x: ", size(x))
	println()
	println(x[:, 1, 1, 1, 1])
	println()
	println("Size x̂: ", size(x̂))
	println()
	println(x̂[:, 1, 1, 1, 1])
end

# ╔═╡ 51d81924-59b1-11eb-2ead-ed5d20387bf5
md"""
## Train model
"""

# ╔═╡ 5782615c-59b1-11eb-360a-456b2589478c


# ╔═╡ Cell order:
# ╟─27add6de-3922-11eb-0955-156a939a344f
# ╟─528e6bb6-3922-11eb-1961-41fc9fb4a094
# ╠═efe8a1fc-5b84-11eb-27bf-23d304283029
# ╠═cd611ea8-4ba4-11eb-025f-2db195ae152b
# ╠═5e89f012-4ba5-11eb-3e35-63a1d62e3e94
# ╟─96121a84-3924-11eb-1cbb-0de4f3c73b0c
# ╠═ef664078-3925-11eb-3b3d-438868228269
# ╠═3ebe6c74-3926-11eb-2a01-f951f7a6849c
# ╠═4a8101aa-3926-11eb-1d4b-0721530a9423
# ╠═57fd3a7e-3926-11eb-0e3f-858f616cea60
# ╟─0ea7faf0-4acc-11eb-2ffd-7bf66092c013
# ╠═cd647910-4acb-11eb-2ee7-972c711247dc
# ╟─99e83cd8-3929-11eb-0a0e-e1d5780c77b4
# ╠═43d816ce-4a40-11eb-1647-f37930d8eb8a
# ╠═797e08be-4a41-11eb-0c4e-1df9d65735b6
# ╠═ccbd3738-53dc-11eb-27cb-4daae45f3216
# ╠═794f97cc-4a41-11eb-08f6-e1487743247d
# ╟─893b0544-4ace-11eb-37d7-6da2eb2ec12d
# ╠═3c77c182-4acb-11eb-22c5-9f100c192b0d
# ╟─96aef858-53dd-11eb-2b07-d17a14c76ed5
# ╠═14ca4132-53e5-11eb-2027-75733a1fac04
# ╟─552d6304-53e4-11eb-125b-c18befd4ba5e
# ╟─9eaf316c-53dd-11eb-0d26-9f045503a6b6
# ╟─029d10ca-53e5-11eb-329b-f73a2fbce197
# ╟─62148508-3926-11eb-34d2-a7bedab25f30
# ╠═c01c2eb6-4acf-11eb-2a85-2fe31d094684
# ╟─9ea1d47e-53e5-11eb-0ab5-117b5c7a8422
# ╠═c59a49de-5889-11eb-14a7-afabd132719b
# ╟─f653f67e-588c-11eb-01d4-5f3c896ed5b2
# ╠═bc5228ce-5891-11eb-29eb-e9bbf5ed0510
# ╟─480897d2-588c-11eb-2894-699a81fc1358
# ╠═9ea0f536-579b-11eb-1709-95271e9a775e
# ╟─32e76378-588d-11eb-0cfe-79706afc57f4
# ╠═1ccd9a18-5889-11eb-3a76-b10e73357829
# ╟─b5b7abd2-588d-11eb-17f0-8dd8860d441a
# ╠═e18b2054-588d-11eb-30a3-3b339c1370b3
# ╟─e8d40100-588d-11eb-16dc-77fd936d9f44
# ╠═3474dc06-5889-11eb-28a6-b19e3b392af8
# ╟─1a5d9a88-588e-11eb-2b0c-3115e94329db
# ╟─4f019f5e-5894-11eb-0810-d343b6345fc6
# ╟─d4ff40a4-5b85-11eb-1216-ed4c6029426c
# ╟─5acdb270-5b85-11eb-3590-afaf2ac61d88
# ╟─51d81924-59b1-11eb-2ead-ed5d20387bf5
# ╠═5782615c-59b1-11eb-360a-456b2589478c
