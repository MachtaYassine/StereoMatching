# StereoMatching
Experimenting with stereo matching



## File and folder structure

- data folder: contains images and depth results
- Stereo folder contains my attmpt at trying to organize the "taxonomy" of stereo in classes for an easy pipeline (It's rushed)
    - Costs: different cost functions
    - Aggregation: only fixed window for now
    - Disparity computing : the different global and energy minimozation algorith besides WTA
    - Disparity refinment: nothing yet
- utilities : data loading and visulisation
## Acknowlegments

- I was greatly inspired by https://github.com/2b-t/stereo-matching for his jit implementation but wasn't able to replicate it well