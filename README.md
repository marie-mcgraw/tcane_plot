# tcane_plot
![overview of TCANE model](TCANE_overview.png)
This repository contains the real-time visualization code for the Tropical Cyclone Artificial Neural-network Error model (TCANE). After you have cloned this repo using `git clone`, navigate to the correct directory (`tcane_plot`). 
# Compatibility Notes
This code requires `libstdc++.so.6 version >= 3.4.29`.

Before you set up the environment, make sure that you have the correct version of the GCC's C++ compiler. 

To find the version installed on your system, run 

`strings /usr/lib64/libstdc++.so.6 | grep ^GLIBCXX_3.4.2`

To find the version installed in your Anaconda or Miniconda, run 

`strings /home/your-username/path-to-conda/anaconda3/lib/libstdc++.so.6 | grep ^GLIBCXX_3.4.2`

To use the latest version of `GLIBCXX`, add `LD_LIBRARY` path to your `.bashrc`. That can be done with the command: 

`echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/miniconda3/lib'>>~/.bashrc`

After you've ensured that you are using a modern enough C++ compiler, proceed to building the Python environment

# Creating the Python Environment
1. Set up the environment file using `mamba` (preferred) or `conda`:
  
`mamba create -f environment.yml` (for `mamba`), or

`conda env create -f environment.yml` for `conda`

 Note that creating the environment might take a few minutes; don't panic!). This is the same environment as the `tcane_intensity` and `tcane_track` code used to run the TCANE model (see Martin Fernandez's [intensity](https://github.com/mafern/tcane_intensity) and [track](https://github.com/mafern/tcane_track) repos for the most up-to-date version of TCANE). So if you've already run the TCANE training code, you should be able to use the same environment here.

2. To activate environment, run `mamba activate env-hurr-tfp` or `conda activate env-hurr-tfp`, depending on which you are using. To deactivate, the command is `mamba deactive, or `conda deactivate`.

## Environment Troubleshooting
There are a few packages that can be troublesome. If you're running into issues building the environment or running the code, here are a couple of packages we've noticed to be issues in the past:
* Double check your versions of `cartopy` and `shapely` by using `conda list cartopy` and `conda list shapely`. `cartopy` should be version 0.21.0 and `shapely` should be version 1.8.2. If needed, reinstall the correct versions of these packages (e.g., `pip install cartopy==0.21.0`).
	* Double check that `matplotlib` is version 3.7.x (`conda list matplotlib`). 
	* Finally, downgrade `protobuf` to 3.19.x. We recommend 3.19.6 (e.g., `pip install protobuf==3.19.6`)

# Code Seupt
1. Run the setup script to create appropriate directories: `python setup.py`
2. The plotting code can only run if you have already created realtime TCANE input files and output files (and TCANE climatology). The plotting code can be run from the command line with the following inputs: `python run_plotting_code.py STORMDATE INPUT_DIR OUTPUT_DIR BDECK_DIR`
  * `STORMDATE` contains the date and location of the storm to plot. It should have the format `BBNNYYYY_MMDDHH`, where `BB` is the 2-letter basin abbreviation and `NN` is the 2 character storm number.
  * `INPUT_DIR` is the directory where the TCANE input files are located on your machine.
  * `OUTPUT_DIR` is the directory where the TCANE output files are located on your machine.
  * `BDECK_DIR` is the directory where bdeck and edeck information is located on your machine. 
3. Figures will be saved in `Figures/DATE/`, in both PNG and PDF format. Log files will also be written. From here, the images can be posted to real-time forecasting webpages, such as CIRA's tc-realtime.
4. Some examples of our visual evaluation tools:
![TCANE category probability](tcane_catprob.png) **Figure 1**. Probability of Hurricane Helene reaching Category 1, 2, 3, 4, and 5 strength for forecasts issued on 09-24-2024 12Z.
![TCANE track predictions](tcane_track.png) **Figure 2**. Track error forecasts for Hurricane Helen issued on 09-24-2024 12Z. Solid orange lines indicate the TCANE error forecasts for the 66.67%-ile, while dashed lines indicate the climatological forecasts at the 66.67%-ile. The 66.67%-ile is chosen to best match the existing National Hurricane Center forecast uncertainty. 
![TCANE rapid intensification](TCANE_rapid_intensification.png) **Figure 3**. Skill of TCANE rapid intensification predictions versus other statistical models.

Stay tuned! TCANE will be running in real time on the CIRA TC Realtime webpage for the 2025 tropical cyclone season! 

# Authors
[Marie C. McGraw](https://github.com/marie-mcgraw) and [Martin A. Fernandez](https://github.com/mafern), updated 2025. 

# References
[1] Barnes, Elizabeth A., Randal J. Barnes and Nicolas Gordillo, 2021: Adding Uncertainty to Neural Network Regression Tasks in the Geosciences, arXiv 2109.07250.

[2] Barnes, Elizabeth A., Randal J. Barnes and Mark DeMaria, 2022: Sinh-arcsinh-normal distributions to add uncertainty to neural network regression tasks: applications to tropical cyclone intensity forecasts, preprint available at https://doi.org/10.31223/X51649.

[3] Fernandez, M.A., Elizabeth A. Barnes, Randal J. Barnes, Mark DeMaria, Marie McGraw, Galina Chirokova, and Lixin Lu, 2025: Predicting Tropical Cyclone Track Forecast Errors using a Probabilistic Neural Network. *Artificial Intelligence for the Earth Systems*, **4**, https://doi.org/10.1175/AIES-D-24-0066.1. 

# License
This repository is licensed under an MIT license.
