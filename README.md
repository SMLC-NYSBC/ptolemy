# Ptolemy
A Python machine learning and computer vision library for automating cryo-EM data collection. The accompanying paper is published at [IUCrJ](https://journals.iucr.org/m/issues/2023/01/00/pw5021/index.html). Ptolemy is designed to handle localization and scoring of squares in low-mag images (pixelsize of around 2000-5000 Angstroms/pixel) and holes in medium-mag images (100-1000 angstroms/pixel). It works on both gold and carbon holey, untilted grids.

Ptolemy has been incorporated into the Leginon screening software, termed Smart-Leginon Autoscreen - paper [here](https://journals.iucr.org/m/issues/2023/01/00/eh5015/). It enables unattended grid screening, reducing required operator time from ~6hrs/grid to just 10 minutes for a cassette. It is on track to save ~1000hrs of operator time at the SEMC across two Glacios's. A tutorial for setting up smart leginon autoscreen on your microscope control computer is available [here](https://emg.nysbc.org/redmine/projects/leginon/wiki/Multi-grid_autoscreening).

The current package detects and ranks squares and holes in low and medium mag images as below using fixed, pretrained computer vision models. Active, on-the-fly learning, enabling the microscope to adapt to new grids dynamically and thereby perform fully-automated data collection (to obtain high-magnification micrographs for high-resolution structures), is undergoing real-world testing at the SEMC.

<details><summary>Example Low Mag Image</summary><p>

  <img src="example_images/for_readme/lowmag.png" width="500">
  <img src="example_images/for_readme/lowmag_processed.png" width=500>
  
</p></details>

<details><summary>Example Med Mag Image</summary><p>

  <img src="example_images/for_readme/medmag.png" width="500">
  <img src="example_images/for_readme/medmag_processed.png" width=500>
  
</p></details>

## Functionality
Images and visualization are handled by the `Exposure` class in `ptolemy/images.py`, with algorithms for processing low and medium mag images in `ptolemy/algorithms.py`. The workflow is outlined in the tutorial notebooks. 

## Future
We plan to improve Ptolemy with active learning on individual data collection sessions, support for tilted grids, and superresolution (unbinned) medium-mag images.

## Dependencies
Tested with python 3.9

- pytorch
- torchvision
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- scikit-image

## License
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## Citation
If you use Ptolemy in your research, please consider citing our paper:

```bibtex
@article{kim2023learning,
  title={Learning to automate cryo-electron microscopy data collection with Ptolemy},
  journal={IUCrJ},
  volume={10},
  number={1},
  year={2023},
  publisher={International Union of Crystallography}
}
