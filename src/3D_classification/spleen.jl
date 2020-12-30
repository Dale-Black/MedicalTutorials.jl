### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ fbc8136c-4acf-11eb-25d0-11a3337582fd
using Pkg

# ╔═╡ e6f4e9d8-4acf-11eb-3cb8-435c7faedeb0
using NIfTI: niread

# ╔═╡ ebf9d894-4acf-11eb-25e3-03c593a7dab8
using LearnBase

# ╔═╡ d1d57786-4acf-11eb-32e1-553bff8011f6
using Glob: glob

# ╔═╡ d56fc3cc-4acf-11eb-1656-1d50718dd5ab
using MLDataPattern: splitobs, getobs

# ╔═╡ d8440a72-4acd-11eb-3375-29e40c34f75e
using DLPipelines

# ╔═╡ b7c8882a-4acc-11eb-15b8-2f4447aa8071
using DataAugmentation

# ╔═╡ b64d8310-4acf-11eb-3e9d-15be083ee7de
using Random: seed!

# ╔═╡ 27add6de-3922-11eb-0955-156a939a344f
md"""
## Spleen 3D Segmentation with Julia and FluxTraining.jl
This tutorial shows how to use Julia and FluxTraining.jl to train a model to segment the spleen in a publicly available CT dataset. Adapted from [link](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/spleen_segmentation_3d.ipynb)
"""

# ╔═╡ 528e6bb6-3922-11eb-1961-41fc9fb4a094
md"""
## Set up environment
"""

# ╔═╡ 00131674-4ad0-11eb-34f9-a5d810992568
Pkg.activate(".")

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
	
	
	LearnBase.nobs(ds::imagesTr) = length(ds.files)
	LearnBase.getobs(ds::imagesTr, idx::Int) = niread(ds.files[idx]).raw


	LearnBase.nobs(ds::labelsTr) = length(ds.files)
	LearnBase.getobs(ds::labelsTr, idx::Int) = niread(ds.files[idx]).raw
	
	
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
		x, y = getobs(train_files, 1)
		typeof(x), size(x), size(y)
	end
	
	
end

# ╔═╡ 99e83cd8-3929-11eb-0a0e-e1d5780c77b4
md"""
## Set up data processing
"""

# ╔═╡ 43d816ce-4a40-11eb-1647-f37930d8eb8a
begin
	
	
	abstract type ImageSegmentationTask <: DLPipelines.Task end
	
	struct ImageSegmentationSimple <: DLPipelines.Method{ImageSegmentationTask}
		imagesize
	end
	
	method = ImageSegmentationSimple((96, 96, 96))
	
end;

# ╔═╡ 797e08be-4a41-11eb-0c4e-1df9d65735b6
begin
	
	
	
	function DLPipelines.encodeinput(
			method::ImageSegmentationSimple,
			context::Training,
			image)
		tfm = RandomResizeCrop(method.imagesize) |> NormalizeIntensity()
		return apply(tfm, image) |> itemdata
	end
	
	function DLPipelines.encodeinput(
			method::ImageSegmentationSimple,
			context::Validation,
			image)
		tfm = CenterResizeCrop(method.imagesize) |> NormalizeIntensity()
		return apply(tfm, image) |> itemdata
	end
	

end

# ╔═╡ 794f97cc-4a41-11eb-08f6-e1487743247d
begin
	
	
	methoddata_train = MethodDataset(train_files, method, Training())
	methoddata_valid = MethodDataset(val_files, method, Validation())


end

# ╔═╡ 893b0544-4ace-11eb-37d7-6da2eb2ec12d
md"""
Double check that the data processing works as expected. After applying the transforms, the first images `(x, y)` should go from size == `(512, 512, 55)` to size = `(96, 96, 96)`
"""

# ╔═╡ 3c77c182-4acb-11eb-22c5-9f100c192b0d
begin
	

	let
		x, y = getobs(methoddata_train, 1)
		size(x) == (96, 96, 96)
		size(y) == (96, 96, 96)
	end
	
end

# ╔═╡ 62148508-3926-11eb-34d2-a7bedab25f30
md"""
## Set deterministic training for reproducibility
"""

# ╔═╡ c01c2eb6-4acf-11eb-2a85-2fe31d094684
seed!(1);

# ╔═╡ Cell order:
# ╟─27add6de-3922-11eb-0955-156a939a344f
# ╟─528e6bb6-3922-11eb-1961-41fc9fb4a094
# ╠═fbc8136c-4acf-11eb-25d0-11a3337582fd
# ╠═00131674-4ad0-11eb-34f9-a5d810992568
# ╟─96121a84-3924-11eb-1cbb-0de4f3c73b0c
# ╠═ef664078-3925-11eb-3b3d-438868228269
# ╠═3ebe6c74-3926-11eb-2a01-f951f7a6849c
# ╠═e6f4e9d8-4acf-11eb-3cb8-435c7faedeb0
# ╠═ebf9d894-4acf-11eb-25e3-03c593a7dab8
# ╠═4a8101aa-3926-11eb-1d4b-0721530a9423
# ╠═d1d57786-4acf-11eb-32e1-553bff8011f6
# ╠═d56fc3cc-4acf-11eb-1656-1d50718dd5ab
# ╠═57fd3a7e-3926-11eb-0e3f-858f616cea60
# ╟─0ea7faf0-4acc-11eb-2ffd-7bf66092c013
# ╠═cd647910-4acb-11eb-2ee7-972c711247dc
# ╟─99e83cd8-3929-11eb-0a0e-e1d5780c77b4
# ╠═d8440a72-4acd-11eb-3375-29e40c34f75e
# ╠═43d816ce-4a40-11eb-1647-f37930d8eb8a
# ╠═b7c8882a-4acc-11eb-15b8-2f4447aa8071
# ╠═797e08be-4a41-11eb-0c4e-1df9d65735b6
# ╠═794f97cc-4a41-11eb-08f6-e1487743247d
# ╟─893b0544-4ace-11eb-37d7-6da2eb2ec12d
# ╠═3c77c182-4acb-11eb-22c5-9f100c192b0d
# ╟─62148508-3926-11eb-34d2-a7bedab25f30
# ╠═b64d8310-4acf-11eb-3e9d-15be083ee7de
# ╠═c01c2eb6-4acf-11eb-2a85-2fe31d094684
