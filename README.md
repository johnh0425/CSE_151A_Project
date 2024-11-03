# CSE 151A Group Project <a target="_blank" href="https://colab.research.google.com/github/brandoluu/CSE_151A_Project/blob/main/CSE151A_Project.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>

## Milestone 2

### Initial Data Exploration:
The dataset contains 16 different subdirectories, each containing different audio samples. There is a total of 176,581 entries in the dataset, and upon looking at the dataset we noticed that many of the files were corrupted. Using the `pretty-midi` utility functions, we can extract the different attributes of the dataset. For the purposes of this project, we will be focusing on 3 components: 
-[1] `n_instruments`: the number of unique instruments in the file
-[2] `tempo_estimate`: the esimated tempo of the file
-[3] `program_numbers`: the unique integer value corresponding to the type of instrument used in the file. The list of program numbers can be found [here](https://midiprog.com/program-numbers/)

### Preprocessing:
As mentioned in the data exploration section we have noticed that a lot of the entries contain corrupted audio files. We have decided to drop these entries as the dataset already contains a lot of entries. 
