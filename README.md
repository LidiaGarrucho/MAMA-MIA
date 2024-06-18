![Header](docs/_static/logo_header.png)
# MAMA-MIA: A Large-Scale Multi-Center Breast Cancer DCE-MRI Benchmark Dataset with Expert Segmentations
[![arXiv](https://img.shields.io/badge/arXiv-XXXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXXXX) [![Synapse](https://img.shields.io/badge/Synapse-syn60868042-1258.svg)](https://doi.org/10.7303/syn60868042)
 
Welcome to the **MAMA-MIA** dataset repository! This dataset contains **1,506** cases of **breast cancer** dynamic contrast-enhanced magnetic resonance images (**DCE-MRI**) with **expert tumor segmentations**. Below, you will find all the necessary information to download and use the dataset, as well as instructions on how to run inference using our pre-trained nnUNet model.

## Content
![Dataset Description](docs/_static/dataset_info.png)

## Expert Segmentations
![Expert Segmentation](docs/_static/segmentation_process.png)

## Downloading the Dataset
The MAMA-MIA dataset is hosted on Synapse ([https://doi.org/10.7303/syn60868042](https://doi.org/10.7303/syn60868042)).
You can download the dataset using the CLI or Python with the following code:

##### Command Line Interface (CLI)
```bash
synapse get syn60868042
```

##### Python

```bash
import synapseclient
syn = synapseclient.Synapse()
syn.login()
entity = syn.get("syn60868042")
```
> Check Synapse [documentation](https://help.synapse.org/docs/Downloading-Data-Programmatically.2003796248.html) for more info. 

## Running Inference with nnUNet pre-trained model
The pre-trained vanilla nnUNet model has been trained using the 1506 full-image DCE-MRIs and the expert segmentations from the MAMA-MIA dataset. 

##### Step 1. Clone the repository
Clone the forked repository of nnUNet.
````
git clone https://github.com/LidiaGarrucho/MAMA-MIA
cd MAMA-MIA/nnUNet
````

##### Step 2. Install the necessary dependencies
To run the pre-trained nnUNet model, follow the [installation instructions](https://github.com/LidiaGarrucho/nnUNet/blob/master/documentation/installation_instructions.md).

##### Step 3. Download the pre-trained weights
The nnUNet pretrained weights can be dowloaded from [Synapse](https://www.synapse.org/Synapse:syn61247992).
Unzip the folder inside nnUNet GitHub repository under `nnUNet/nnunetv2/nnUNet_results`.

##### Step 4. (Recommended) Preprocess your input MRI images
The recommended preprocessing steps to get optimum performance are:
- **z-score normalization**. For DCE-MRI, use the mean and standard deviation of all the phases (from pre to last post-contrast) to z-score the DCE-MRI sequence.
- **isotropic pixel spacing**. The MRIs were resampled using a uniform pixel spacing of [1,1,1].

##### Step 5. Run the nnUNet inference:!
````
nnUNetv2_predict -i /path/to/your/images -o /path/to/output -d 101 -c 3d_fullres
````
- Replace `/path/to/your/images` with the directory containing your input images.
- Replace `/path/to/output` with the directory where you want to save the output segmentations.

> Note: An error might arise if your images are not in compressed NifTI format (.nii.gz).

## Reference and License
If you use the MAMA-MIA dataset or the pretrained model in your research, please cite our publication and the dataset publications of the images included in the dataset.

### MAMA-MIA Dataset Citation
> Placeholder for ArXiV paper citation

BibTex entry:
````
@article{lidia2024mamamia,
  title={},
  author={},
  journal={},
  year={2024},
  publisher={}
}
````

The MAMA-MIA dataset includes public DCE-MRI images from four different collection in the TCIA repository under the following licenses:

#### ISPY1 Trial (License [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)) 

> David Newitt, Nola Hylton, on behalf of the I-SPY 1 Network and ACRIN 6657 Trial Team. (2016). Multi-center breast DCE-MRI data and segmentations from patients in the I-SPY 1/ACRIN 6657 trials. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.HdHpgJLK](https://www.cancerimagingarchive.net/collection/ispy1/)

#### Breast-MRI-NACT-Pilot (License [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)) 
> Newitt, D., & Hylton, N. (2016). Single site breast DCE-MRI data and segmentations from patients undergoing neoadjuvant chemotherapy (Version 3) [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/K9/TCIA.2016.QHsyhJKy](https://doi.org/10.7937/K9/TCIA.2016.QHsyhJKy)

#### ISPY2 Trial (License [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) 

> Li, W., Newitt, D. C., Gibbs, J., Wilmes, L. J., Jones, E. F., Arasu, V. A., Strand, F., Onishi, N., Nguyen, A. A.-T., Kornak, J., Joe, B. N., Price, E. R., Ojeda-Fournier, H., Eghtedari, M., Zamora, K. W., Woodard, S. A., Umphrey, H., Bernreuter, W., Nelson, M., … Hylton, N. M. (2022). I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2)  (Version 1) [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/TCIA.D8Z0-9T85](https://doi.org/10.7937/TCIA.D8Z0-9T85)

#### Duke-Breast-Cancer-MRI (License [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))
> Saha, A., Harowicz, M. R., Grimm, L. J., Weng, J., Cain, E. H., Kim, C. E., Ghate, S. V., Walsh, R., & Mazurowski, M. A. (2021). Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations [Data set]. The Cancer Imaging Archive. [https://doi.org/10.7937/TCIA.e3sv-re93](https://doi.org/10.7937/TCIA.e3sv-re93)

![Collaborators](docs/_static/collaborators.png)

Thank you for using our dataset and pretrained model! If you have any questions or issues, please feel free to open an issue.

