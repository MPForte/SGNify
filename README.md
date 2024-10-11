## Reconstructing Signing Avatars From Video Using Linguistic Priors

[[Project Page](https://sgnify.is.tue.mpg.de/)] 
[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.pdf)]
[[Supp. Mat.](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Forte_Reconstructing_Signing_Avatars_CVPR_2023_supplemental.pdf)]

![Teaser](https://sgnify.is.tue.mpg.de/media/upload/teaser.png)

## Table of Contents
  * [License](#license)
  * [Description](#description)
    * [Setup](#setup)
    * [Fitting](#fitting)
  * [Dependencies](#dependencies)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)


## License

Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the [terms and conditions](https://github.com/MPForte/sgnify/blob/master/LICENSE) and any accompanying documentation before you download and/or use the SGNify model, data and software, (the "Model & Software"), including 3D meshes, blend weights, blend shapes, textures, software, scripts, and animations. By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Description

This repository contains the fitting code used for the experiments in [Reconstructing Signing Avatars From Video Using Linguistic Priors](https://sgnify.is.tue.mpg.de/).

### Setup
1. Create an account at https://sgnify.is.tue.mpg.de/
2. Ensure a latest `conda` version
2. Clone this repo (without `recursive`)
4. Change directory `cd SGNify`
5. Run `./install.sh`

To test that the setup worked, run:
```Shell
conda activate sgnify
python sgnify.py \
    --image_dir_path data/demo/test_frames \
    --output_folder data/demo/output_test
```
We provided the expected output; check that `data/demo/output_test` and `data/demo/output` have the same results.

### Fitting 
Run the following command to execute the code:
```Shell
python sgnify.py \
    --image_dir_path DATA_PATH \
    --output_folder OUTPUT_FOLDER
```
where the `DATA_PATH` should be either a path to a video or a folder of images.

## Citation

If you find this Model & Software useful in your research we would kindly ask you to cite:

```
@InProceedings{Forte_2023_CVPR,
    author    = {Forte, Maria-Paola and Kulits, Peter and Huang, Chun-Hao P. and Choutas, Vasileios and Tzionas, Dimitrios and Kuchenbecker, Katherine J. and Black, Michael J.},
    title     = {Reconstructing Signing Avatars From Video Using Linguistic Priors},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {12791-12801}
}
```

We use [SPECTRE](https://github.com/filby89/spectre) to estimate FLAME parameters. Consider also citing them as:
```
@InProceedings{Filntisis_2023_CVPR,
    author    = {Filntisis, Panagiotis P. and Retsinas, George and Paraperas-Papantoniou, Foivos and Katsamanis, Athanasios and Roussos, Anastasios and Maragos, Petros},
    title     = {SPECTRE: Visual Speech-Informed Perceptual 3D Facial Expression Reconstruction From Videos},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {5744-5754}
}
```

## Acknowledgments

We thank Galina Henz and Tsvetelina Alexiadis for trial coordination; Matvey Safroshkin, Markus Höschle, Senya Polikovsky, Tobias Bauch, Taylor McConnell (TM), and Bernard Javot for the capture setup; TM for data-cleaning coordination; Leyre Sánchez Vinuela, Andres Camilo Mendoza Patino, and Yasemin Fincan for data cleaning; Nima Ghorbani and Giorgio Becherini for MoSh++; Joachim Tesch for help with Blender; Benjamin Pellkofer and Joey Burns for IT support; Yao Feng, Anastasios Yiannakidis, and Radek Danˇeˇcek for discussions on facial methods; Haliza Mat Husin and Mayumi Mohan for help with statistics; and the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting Maria-Paola Forte and Peter Kulits.

## Contact

For questions, please contact [forte@tue.mpg.de](mailto:forte@tue.mpg.de). 

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de).
