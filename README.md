# GPS Data to Comprehensible Racetrack for LapTime Simulation

## Abstract

This project is part of the SD2229 and SD2230 courses in KTH, Stockholm, Sweden. Its purpose is to help the Formula Student team on the Autocross event by having a visual representation of the unknown track, the optimal path for the car to follow and the lap times quickly and efficently.


## Description

Our LapTime Simulator in Matlab takes as an input for the track an Excel file in which data about the lenght of the straight lines, the radius and length of the turns, is stored.

![Excel](https://github.com/arthur1911/SD2230--GPS-to-Racetrack/assets/102413353/ec42a065-0f4c-44c3-baf5-6066c067a83b)

In order to convert GPS data into this kind of file, we first need to record GPS data using Strava or any app like this one. Then, we export the .gpx file and put this as an input in our code.

The program then smooth the data, identify the straight lines using linear regressions and connect those with arcs. A correction is made at the end to have a smooth track.

![GPX_meters](https://github.com/arthur1911/SD2230--GPS-to-Racetrack/assets/102413353/f1a89580-bdc8-4082-bcd2-39595c753f8d)

![GPX_smoothed](https://github.com/arthur1911/SD2230--GPS-to-Racetrack/assets/102413353/f3e16519-9e93-446a-93c6-2976b70662f1)

![straight_lines](https://github.com/arthur1911/SD2230--GPS-to-Racetrack/assets/102413353/b9b80af6-6600-415b-8dd4-ad3ff6fba5b6)

![modelled_straight_lines](https://github.com/arthur1911/SD2230--GPS-to-Racetrack/assets/102413353/5766919a-0cf8-4111-a9f3-2f9302e9c759)

![lines_and_arcs](https://github.com/arthur1911/SD2230--GPS-to-Racetrack/assets/102413353/f8d854b6-b019-407a-add4-6fc2ceb6de2a)

![perfect_track](https://github.com/arthur1911/SD2230--GPS-to-Racetrack/assets/102413353/9354bc27-338d-455d-bdd0-e424a2670655)

The code takes less than a minute to generate the Excel file.

The FSAE team needs to do this process for the inner border of the track and the outer border.

Then, the second program find the optimal path just by using the fact that racecar drivers usually follow the outer border when they are on a straight line and touch the inner border at the midpoints of turns (thoses points are called apexes). The result can be shown below:

![optimal_path](https://github.com/arthur1911/SD2230--GPS-to-Racetrack/assets/102413353/a43fb1bb-060a-42e9-9d15-56cdd254eee2)

We can now use our LapTime Simulator to compute the laptimes for this 1km track.
The car took 31.5 seconds to run on the optimal path whereas it took 33.6 seconds on the inner border and 34.6 seconds on the outer border.

The whole process takes less than 15 minutes to parametrize and have great results.

## How to use it?

Download all the files located in this folder and put them in a folder in your computer.

The process is separated in two parts:
- the creation of the Excel file that represent the track from GPS data (`GPX_to_Excel_2.ipynb` and `functions_GPX_to_Excel.py`)
- the finding of the optimal path for the car to follow (`Find_optimal.py` and `functions_optimal.py`)

For the creation of the Excel, simply run the Jupyter Notebook with your GPS data (one is given here as an example (`Course_pied_dans_l_apr_s_midi.gpx`)) and adjust the parameters to spot correctly the straight lines and the turns. Once you are happy with the result, you will find the Excel in the folder you specified.

For the optimal path finding, you will need to do the previous process twice (one for the inner border of the track and one for the outer). Open the `Find_optimal.py` file, put your 2 .gpx files and adjust the parameters as you want until you are happy with the result. You will find the Excel file of the optimal path in the specified folder.
