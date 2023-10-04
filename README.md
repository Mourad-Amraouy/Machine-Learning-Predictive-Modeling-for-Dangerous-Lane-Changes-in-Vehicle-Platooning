# Machine-Learning-Predictive-Modeling-for-Dangerous-Lane-Changes-in-Vehicle-Platooning
#
Cutting into vehicle platoons represents a hazardous driving maneuver that significantly impacts the safety and efficiency of platooning vehicles. Accurate prediction of such maneuvers plays a pivotal role in enabling platooning systems to proactively implement safety measures that guarantee the well-being and cohesion of the platoon. This project seeks to develop a robust predictive model using the NGSIM dataset, which contains detailed information on vehicle trajectories. By harnessing machine learning techniques and advanced data analysis, our model aims to assess the danger level associated with lane changes, thereby empowering platooning systems to make informed decisions to enhance the overall safety and integrity of the platoon. Through this research, we aim to contribute to the advancement of autonomous driving systems and pave the way for safer and more efficient transportation solutions.
## A. NGSIM dataset
Researchers for the NGSIM program collected detailed vehicle trajectory data on eastbound I-80 in the San Francisco Bay area in Emeryville, CA, on April 13, 2005. The study area was approximately 500 meters (1,640 feet) in length and consisted of six freeway lanes, including a high-occupancy vehicle (HOV) lane. An onramp also was located within the study area. Seven synchronized digital video cameras, mounted from the top of a 30-story building adjacent to the freeway, recorded vehicles passing through the study area. NG-VIDEO, a customized software application developed for the NGSIM program, transcribed the vehicle trajectory data from the video. This vehicle trajectory data provided the precise location of each vehicle within the study area every one-tenth of a second, resulting in detailed lane positions and locations relative to other vehicles.
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://l.top4top.io/p_28331mieg1.png">
  <source media="(prefers-color-scheme: light)" srcset="https://l.top4top.io/p_28331mieg1.png">
  <img alt="visualizing a Right Lane Change Maneuver." src="https://l.top4top.io/p_28331mieg1.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://f.top4top.io/p_28338oqft1.png">
  <source media="(prefers-color-scheme: light)" srcset="https://f.top4top.io/p_28338oqft1.png">
  <img alt="visualizing a Right Lane Change Maneuver." src="https://f.top4top.io/p_28338oqft1.png">
</picture>


## B.The Lane Change Maneuver 
#
A cut-in is described as a lane change executed by a nearby vehicle in an adjacent lane, wherein this vehicle switches lanes and enters the lane in front of the ego vehicle. Depending on the side from which the neighboring vehicle initiates the lane change, we identify two distinct maneuvers: "Left cut-ins," which occur when the neighboring vehicle changes lanes from the left, and "Right cut-ins," which occur when the neighboring vehicle changes lanes from the right. You can observe an example of a right cut-in maneuver in Figure 1, where the white vehicle changes lanes between the two blue vehicles. The responsibility for predicting the cut-in maneuver falls upon the blue vehicle in front of which the white car makes the lane change. Subsequently, the vehicle executing the cut-in is referred to as the "target vehicle," while the one responsible for making the prediction is known as the "host vehicle."

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRV4ZwSVRKyqBdga1yNvSlRsTrcVST6NNUh1VABUwiwArRBnLj-">
  <source media="(prefers-color-scheme: light)" srcset="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRV4ZwSVRKyqBdga1yNvSlRsTrcVST6NNUh1VABUwiwArRBnLj-">
  <img alt="visualizing a Right Lane Change Maneuver: The White Vehicle Inserts Between Two Blue Vehicles." src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRV4ZwSVRKyqBdga1yNvSlRsTrcVST6NNUh1VABUwiwArRBnLj-">
</picture>

1. visualizing a Right Lane Change Maneuver: The White Vehicle Inserts Between Two Blue Vehicles.

#


![I80_Study area schematic related to NGSIM data](https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRrYpAvfuIaz_IBz_VmCVp3mSW7vB00zSpphkzMDuDWQr5WgH4m)

2. I80_Study area schematic related to NGSIM data


#





## C.Cut-in extraction
To prepare the dataset for classification, the initial step involves conducting an analysis to identify instances of cut-in events. Specifically, this study concentrates on cut-ins originating from vehicles in the right adjacent lanes. To automate the process, an algorithm was devised to extract potential cut-ins from the dataset, taking into account the attributes outlined in Table I. A potential cut-in is recognized when a vehicle in the right adjacent lane at timestamp t transitions into a Preceding vehicle in the subsequent timestamp t+1.
_
![Potential interfering vehicle tracking window](https://c.top4top.io/p_2833n257g1.png)
_
#

## Data preparation and engineering for the creation of a dangerous lane change prediction model using the NGSIM dataset

This code aims to prepare the data and perform the necessary engineering to create a prediction model to determine if a lane change is dangerous or not. It uses the NGSIM dataset, which contains information on vehicle trajectories.

`import pandas as pd`

`import numpy as np`

`import sys`

`from tqdm import tqdm`

`from pandarallel import pandarallel`
`import warnings`

### first preparation code
This is a script that aims to find and add the neighbors of each row of data in the "new_df" DataFrame. It uses the "get_neighbors" function to find neighbors based on the specified distance and lane criteria. The results are then added to the "new_df" DataFrame. The script also uses the "pandarallel" library to parallelize the calculation and speed up the process. Warnings are ignored with warnings.filterwarnings('ignore'), and a progress bar is displayed with "tqdm" to track the progress of vehicle processing. Finally, the prepared data is saved to a specified output file.

### add 'Right_cuts', 'Left_cuts'
In this code, the detect_cuts function takes a row of the new_df DataFrame as input, extracts information about neighboring vehicles on the right and left, and returns a list of potential cuts for each side. and the result is stored in two new columns 'Right_cuts' and 'Left_cuts' of the DataFrame. Finally, the updated DataFrame is saved in a CSV file 'trajectories_with_cuts.csv'.

### The describe_cut_in_R_L function
The describe_cut_in_R_L function describes right and left cut-ins in a given DataFrame. Here's how it works:

It initializes the counters s, d, s1, d1, z, and z1 to zero. These counters are used to count different types of cuts.
Lists f1 and f2 contain the names of the columns corresponding to the right and left cuts respectively.
The function iterates through each row of the DataFrame df and counts the number of rows where the right and left cut columns are empty or non-empty.
The counters s, d, s1, and d1 are updated accordingly.
Additionally, the function also counts the number of lines where both the right and left cuts are non-empty (represented by the z counter) and the number of lines where both the right and left cuts are empty (represented by counter z1).
Finally, the function displays the results, showing the number of cuts on the right, the number of non-cuts on the right, the number of cuts on the left, the number of non-cuts on the left, the number of cuts on the right and left, and the number of uncuts on both the right and left.

### Function to find the nearest cut vehicle id
The find_nearest_cut function allows you to find the closest cutting vehicle for each row of the DataFrame df. Here's how it works:

- It retrieves the lists of cuts to the right (right_cuts) and to the left (left_cuts) of the current line.
+ Then it concatenates these two lists to form an all_cuts list containing all the cutting vehicle IDs.
* If the all_cuts list is empty, the function returns NaN, indicating that no vehicles cut the path for this line.
- Otherwise, it iterates over each cut identifier in the all_cuts list and calculates the Euclidean distance between the position of the current vehicle and the position of the corresponding cutting vehicle.
+ Distances are stored in the distances list.
* Using np.argmin(distances), the function finds the index of the minimum distance, which corresponds to the index of the closest cut.
- Finally, the function returns the identifier of the closest cut (nearest_cut_id).
+ The find_nearest_cut function is applied to each row of the df DataFrame using df.apply(find_nearest_cut, axis=1), and the result is saved in a "Target" column.
* If no vehicle cuts the path, the value of the "Target" column will be NaN.
- The updated DataFrame is saved in a CSV file at the specified location.

### Add a Cut_in field
this column depends on the value of the "Target" column, if the value of the "Target" column is empty (represented by an empty string " "). If so, the function returns 0, otherwise it returns 1

### Function to determine if the cut-in is dangerous or not (based on 7 seconds and reaction_time and Safety distance)
Here is an is_dangerous function that determines whether a lane change is dangerous or not. Here's how it works:

- It uses information from the current line to calculate the required safety distance, based on the speed of the host vehicle and the driver's reaction time.
- Then, it looks for cuts (vehicles approaching the current vehicle's lane) in the 7 seconds before the lane change.
- If the target vehicle (vehicle with which there is a risk of collision) is present among the cuts, the function checks the distance between the host vehicle and the cutting vehicle.
- If the distance is less than the safety distance, the function returns 1 (dangerous), otherwise it returns 0 (not dangerous).
- If no cutting vehicle is found in the 7 seconds preceding the change of
  
![I80_Study area schematic related to NGSIM data](https://d.top4top.io/p_2833xebp21.png)


![I80_Study area schematic related to NGSIM data](https://d.top4top.io/p_2833hw4lq1.png)

