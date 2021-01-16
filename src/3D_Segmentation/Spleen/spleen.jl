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
	
	# ] add NIfTI
	# ] add Glob
	# ] add MLDataPattern
	# ] add https://github.com/lorenzoh/DLPipelines.jl
	# ] add https://github.com/lorenzoh/DataAugmentation.jl.git
	# ] add PlutoUI
	# ] add Plots
	
	
	using NIfTI
	using Glob
	using MLDataPattern
	using DLPipelines
	using DataAugmentation
	using Random: seed!
	using PlutoUI
	using Plots
	using Flux
	
	
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
		tfm = RandomResizeCrop(method.imagesize) |> NormalizeIntensity()
		return apply(tfm, Image(image)) |> itemdata
	end
	
	function DLPipelines.encodeinput(
			method::ImageSegmentationSimple,
			context::Validation,
			image)
		tfm = CenterResizeCrop(method.imagesize) |> NormalizeIntensity()
		return apply(tfm, Image(image)) |> itemdata
	end
	

end

# ╔═╡ ccbd3738-53dc-11eb-27cb-4daae45f3216
begin
	
	
	function DLPipelines.encodetarget(
			method::ImageSegmentationSimple,
			context::Training,
			image)
		tfm = RandomResizeCrop(method.imagesize)
		return apply(tfm, MaskBinary(reinterpret(Bool, image))) |> itemdata
	end
	
	function DLPipelines.encodetarget(
			method::ImageSegmentationSimple,
			context::Validation,
			image)
		tfm = CenterResizeCrop(method.imagesize)
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
Double check that the data processing works as expected. After applying the transforms, the first images `(x, y)` should go from size == `(512, 512, 55)` to size = `(512, 512, 96)`
"""

# ╔═╡ 3c77c182-4acb-11eb-22c5-9f100c192b0d
begin

	let
		x, y = getobs(methoddata_train, 1)
		size(x) == (512, 512, 96)
		size(y) == (512, 512, 96)
	end
	
	
end

# ╔═╡ 96aef858-53dd-11eb-2b07-d17a14c76ed5
md"""
## Plot data
"""

# ╔═╡ 14ca4132-53e5-11eb-2027-75733a1fac04
x, y = getobs(methoddata_valid, 1);

# ╔═╡ 552d6304-53e4-11eb-125b-c18befd4ba5e
md"""
$(@bind a Slider(1:96))
"""

# ╔═╡ 9eaf316c-53dd-11eb-0d26-9f045503a6b6
heatmap(x[:, :, a], c = :grays)

# ╔═╡ 029d10ca-53e5-11eb-329b-f73a2fbce197
heatmap(y[:, :, a], c = :grays)

# ╔═╡ 62148508-3926-11eb-34d2-a7bedab25f30
md"""
## Set deterministic training for reproducibility
"""

# ╔═╡ c01c2eb6-4acf-11eb-2a85-2fe31d094684
seed!(1);

# ╔═╡ 9ea1d47e-53e5-11eb-0ab5-117b5c7a8422
md"""
## Create model
* Adapted from [link](https://gist.github.com/haampie/bceb1d59fd9a44f092f913062e58d482)
"""

# ╔═╡ 9ea0f536-579b-11eb-1709-95271e9a775e


# ╔═╡ aa8f509a-5796-11eb-3f42-93555ff8dcda
image = reshape(collect(1:100), (5, 4, 5));

# ╔═╡ Cell order:
# ╟─27add6de-3922-11eb-0955-156a939a344f
# ╟─528e6bb6-3922-11eb-1961-41fc9fb4a094
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
# ╠═9ea0f536-579b-11eb-1709-95271e9a775e
# ╠═aa8f509a-5796-11eb-3f42-93555ff8dcda
