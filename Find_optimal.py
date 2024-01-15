###################### LIBRAIRIES ######################

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import pandas as pd
from sympy import symbols, Eq, solve, atan2, pi
import functions_optimal as fo

###################### PARAMETERS ######################

# Define the path to the CSV files
file_path_outer = 'outer_border.csv'  
file_path_inner = 'inner_border.csv'  

# Define the starting points of the inner and outer borders
inner_start = (0, 0)
outer_start = (0, -10)
opti_start = (0, -10)

# Define the starting angles of the inner and outer borders
inner_start_angle = 0
outer_start_angle = 0

# Define the treshold for the straight lines
threshold = 0.35

# Define the adjustment step for the straight lines
adjustment_step = 0.05

# Define the maximum number of steps for the adjustment of the straight lines
step_max = 20

# Define the path to the CSV file where the optimal path will be saved
path_csv = 'C:/Users/arthu/OneDrive/Documents/ENSTA/KTH/SD2229/GPX to optimal race path/optimal_path.csv'

###################### MAIN CODE ######################

# Read the CSV files into DataFrames and correct the decimal separator
outer_df = pd.read_csv(file_path_outer, sep=';')
inner_df = pd.read_csv(file_path_inner, sep=';')

# Get the tangents and endpoints of the straight lines
tangents_outer, line_endpoints_outer = fo.get_tangents_and_endpoints(outer_df, outer_start, outer_start_angle)

# Adjust the length of the straight lines
line_endpoints_opti, arc_midpoints_opti = fo.adjust_straight_lengths(outer_df, inner_df, inner_start, inner_start_angle, outer_start, outer_start_angle, threshold, adjustment_step, step_max)

# Get the arc parameters
arcs_end, radius, length_arc, arcs = fo.compute_arcs(tangents_outer, line_endpoints_opti)

# Plot the arcs
for i, arc in enumerate(arcs):
    plt.plot(arc[0], arc[1], color='orange')

# Plot the corrected straight lines
length_straight = fo.plot_correct_straight_lines(tangents_outer, line_endpoints_opti, arcs_end)
plt.title('Modeled Racetrack with correct straight lines')
plt.xlabel('Meter')
plt.ylabel('Meter')
plt.axis("equal")
plt.grid(True)
plt.show()

# Save the racetrack to a CSV file
fo.save_dataframe_to_csv(tangents_outer, length_straight, length_arc, radius, path_csv)

# Create a figure and a single axis
fig, ax = plt.subplots()

# Read the CSV file into a DataFrame and correct the decimal separator
file_path_opti = 'optimal_path.csv'  
opti_df = pd.read_csv(file_path_opti, sep=';')

# Plot the racetracks on the same axis
fo.plot_racetrack_oriented(inner_start, inner_df, ax, color='black', label='Inner Border')  # Inner track in black
fo.plot_racetrack_oriented(outer_start, outer_df, ax, color='orange', label='Outer Border')  # Outer track in orange
fo.plot_racetrack_oriented(outer_start, opti_df, ax, color='blue', label='Optimal Track') # Optimal track in blue

# Display the legend
plt.legend()

# Show the plot
plt.show()