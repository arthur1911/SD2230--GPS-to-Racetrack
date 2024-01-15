import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sympy import symbols, Eq, solve, atan2, pi
from itertools import cycle
import matplotlib.patches as patches

# Function to calculate the angle between two lines
def calculate_angle(m1, m2):
    # Angle between lines = arctan |(m2 - m1) / (1 + m1*m2)|
    return atan2(abs(m2 - m1), (1 + m1*m2))

# Function to select the correct bisector based on the angle
def select_center(m1, m2, I, red_line_eq, point, A):
    # Define symbols
    x, y = symbols('x y')

    m_bisector1 = (m1 * (1 + m2**2)**0.5 + m2 * (1 + m1**2)**0.5) / ((1 + m2**2)**0.5 + (1 + m1**2)**0.5)
    m_bisector2 = (m1 * (1 + m2**2)**0.5 - m2 * (1 + m1**2)**0.5) / ((1 + m2**2)**0.5 - (1 + m1**2)**0.5)
    
    # Equation of the bisector
    bisector_eq1 = Eq(y - I[y], m_bisector1 * (x - I[x]))
    bisector_eq2 = Eq(y - I[y], m_bisector2 * (x - I[x]))

    # Equation of the perpendicular to the blue line through point A
    # The slope of this line will be the negative reciprocal of the blue line's slope
    if m1 == 0:  # the line is horizontal
        perpendicular_eq = Eq(x, A[0])  # the perpendicular line is vertical
    else:
        perpendicular_slope = -1/m1
        b = A[1] - perpendicular_slope * A[0]  # calculate the y-intercept
        perpendicular_eq = Eq(y, perpendicular_slope * x + b)
    
    # Solve for the intersection point C of the perpendicular line with the bisector to get the center of the circle
    C1 = solve((perpendicular_eq, bisector_eq1), (x, y))
    C2 = solve((perpendicular_eq, bisector_eq2), (x, y))
    
    # Equation of the perpendicular to the red line through point C
    # The slope of this line will be the negative reciprocal of the red line's slope
    perpendicular_slope2 = -1/m2
    perpendicular_eq2_1 = Eq(y - C1[y], perpendicular_slope2 * (x - C1[x]))
    perpendicular_eq2_2 = Eq(y - C2[y], perpendicular_slope2 * (x - C2[x]))
    
    # Solve for the intersection point B of the perpendicular line with the red line
    B1 = solve((perpendicular_eq2_1, red_line_eq), (x, y))
    B2 = solve((perpendicular_eq2_2, red_line_eq), (x, y))
    
    if ((B1[x]-point[0])**2 + (B1[y]-point[1])**2)**(0.5) > ((B2[x]-point[0])**2 + (B2[y]-point[1])**2)**(0.5):
        return C2, B2
    else:
        return C1, B1


def select_arc(A, B, C, radius):
    # Define symbols
    x, y = symbols('x y')

    # Calculate angles for A and B with respect to the center C
    angle_A = atan2(A[1] - C[y], A[0] - C[x])
    angle_B = atan2(B[y] - C[y], B[x] - C[x])
    
    # Adjust angles to range [0, 2*pi]
    angle_A = angle_A % (2 * np.pi)
    angle_B = angle_B % (2 * np.pi)

    # Determine the smaller angle difference
    angle_diff = np.abs(angle_B - angle_A)
    if angle_diff > np.pi:
        angle_diff = 2 * np.pi - angle_diff
    
    # Sort angles to ensure proper arc direction
    if float(angle_B) - float(angle_A) > np.pi:
        angle_B = angle_B - 2*pi
    if float(angle_A) - float(angle_B) > np.pi:
        angle_A = angle_A - 2*pi
        
    # Create an array of angles for the arc
    arc_angles = np.linspace(float(angle_A), float(angle_B), 100)
    angle_arc = angle_diff
    
    # Coordinates for the arc
    arc_x = C[x] + radius * np.cos(arc_angles)
    arc_y = C[y] + radius * np.sin(arc_angles)
    
    return arc_x, arc_y, angle_arc

def find_arc_midpoints(arcs):
    arc_midpoints = []

    for arc_x, arc_y in arcs:
        # Find the middle index of the arc points
        midpoint_index = len(arc_x) // 2

        # Get the midpoint coordinates
        midpoint_x = arc_x[midpoint_index]
        midpoint_y = arc_y[midpoint_index]

        arc_midpoints.append((midpoint_x, midpoint_y))

    return arc_midpoints

def get_tangents_and_endpoints(track_df, start_point, start_angle):
    # Initialize lists to store the data
    df_radius = []
    length_arc = []
    arcs_end = []
    line_endpoints = []
    tangents = []

    # Initialize the current position and angle
    current_x, current_y = start_point[0], start_point[1]
    current_angle = start_angle  

    # Create a list of tuples containing the start and end points of each straight line as well as the slope and intercept
    for index, section in track_df.iterrows():
        if section['Type'] == 'Straight':
            # Calculate end point of the straight line
            dx = section['Section Length'] * np.cos(np.deg2rad(current_angle))
            dy = section['Section Length'] * np.sin(np.deg2rad(current_angle))
            line_endpoints.append(((current_x, current_y), (current_x + dx, current_y + dy)))  # Store as a tuple
            tangents.append((np.tan(np.deg2rad(current_angle)), current_y - np.tan(np.deg2rad(current_angle))*current_x))
            current_x += dx
            current_y += dy
        else:
            # Plot an arc
            df_radius = section['Corner Radius']
            length = section['Section Length']
            angle_change = np.rad2deg(length / df_radius)

            if section['Type'] == 'Left':
                arc_start_angle = current_angle - 90
                arc_end_angle = arc_start_angle + angle_change
                arc_center_x = current_x - df_radius * np.sin(np.deg2rad(current_angle))
                arc_center_y = current_y + df_radius * np.cos(np.deg2rad(current_angle))
            else: # Assuming 'Right' for any non-'Left' type
                arc_start_angle = current_angle - angle_change
                arc_end_angle = current_angle
                arc_center_x = current_x - df_radius * np.sin(np.deg2rad(arc_end_angle))
                arc_center_y = current_y + df_radius * np.cos(np.deg2rad(arc_end_angle))

            current_angle = arc_end_angle 
            current_x = arc_center_x + df_radius * np.cos(np.deg2rad(current_angle))
            current_y = arc_center_y + df_radius * np.sin(np.deg2rad(current_angle))
            current_angle = current_angle + 90
    return tangents, line_endpoints

# Compute the different arcs and plot them
def compute_arcs(tangents, line_endpoints):
    # Define symbols
    x, y = symbols('x y')

    arcs = []
    arcs_end = []
    radius = []
    length_arc = []

    for i in range(len(tangents) - 1):
        # Equation of the blue line
        blue_line_eq = Eq(y, tangents[i][0]*x + tangents[i][1])
        
        # Equation of the red line
        red_line_eq = Eq(y, tangents[i+1][0]*x + tangents[i+1][1])

        # And the point A on the blue line
        A = line_endpoints[i][1]
        
        # Solve for the intersection point B of the blue line with the red line
        I = solve((blue_line_eq, red_line_eq), (x, y))
        
        # Slopes of the blue and red lines
        m_blue = tangents[i][0]
        m_red = tangents[i+1][0]  
        
        # Compute angle between the two lines
        alpha = calculate_angle(m_blue, m_red)
        
        # Get the center of the circle and the intersection with the red line
        C, B = select_center(m_blue, m_red, I, red_line_eq, line_endpoints[i+1][0], A)
        arcs_end.append((B[x], B[y]))
        
        # The radius is the distance between A and C
        radii = ((C[x]-A[0])**2 + (C[y]-A[1])**2)**(1/2)
        radius.append(float(radii))
        
        arc_x, arc_y, angle_arc = select_arc(A, B, C, radii)
        arcs.append((arc_x, arc_y))
        length_arc.append(float(angle_arc)*float(radii))

    return arcs_end, radius, length_arc, arcs
    
def plot_straight_lines(tangents, line_endpoints):
    # Generate a list of distinct colors
    distinct_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'pink', 'lime', 'navy']
    color_cycler = cycle(distinct_colors)

    # Assign a unique color to each straight line for plotting
    line_colors = {i: next(color_cycler) for i in range(len(line_endpoints))}
    # Plot the modeled straight lines
    for i, line_points in enumerate(line_endpoints):
        
        # Plot the line from the start to the end of the straight section
        start_point = line_points[0]
        end_point = line_points[1]
        
        plt.plot([start_point[0], end_point[0]], [tangents[i][0]*start_point[0] + tangents[i][1], tangents[i][0]*end_point[0] + tangents[i][1]], 
                color=line_colors[i], linewidth=2, zorder=2)
        
def is_midpoint_above_threshold(opti_midpoint, inner_midpoint, outer_midpoint, threshold, k):
    """
    Check if the relative distance between the optimal midpoint and the inner midpoint
    is above a threshold percentage of the distance between the outer midpoint and the inner midpoint.
    """
    vec_inner_to_outer = np.array(outer_midpoint) - np.array(inner_midpoint)
    vec_inner_to_opti = np.array(opti_midpoint) - np.array(inner_midpoint)
    relative_distance = np.linalg.norm(vec_inner_to_opti) / np.linalg.norm(vec_inner_to_outer)
    print("Turn n°", k, "relative distance =", relative_distance)
    return relative_distance > threshold

def are_midpoints_above_threshold(arc_midpoints_opti, arc_midpoints_inner, arc_midpoints_outer, threshold):
    """
    Check if all distances between the optimal midpoints and the inner midpoints are above a threshold.
    """
    for k, (opti_midpoint, inner_midpoint, outer_midpoint) in enumerate(zip(arc_midpoints_opti, arc_midpoints_inner, arc_midpoints_outer)):
        if is_midpoint_above_threshold(opti_midpoint, inner_midpoint, outer_midpoint, threshold, k):
            return True
    return False

        
def adjust_straight_lengths(outer_df, inner_df, start_point_inner, start_angle_inner, start_point_outer, start_angle_outer, threshold, adjustment_step, step_max):
    # Get the start and end points of the straight lines
    outer_tangents, outer_line_endpoints = get_tangents_and_endpoints(outer_df, start_point_outer, start_angle_outer)
    inner_tangents, inner_line_endpoints = get_tangents_and_endpoints(inner_df, start_point_inner, start_angle_inner)

    # Initialize the percentages for each straight line
    percentages = [adjustment_step for i in range(len(outer_line_endpoints))]

    # Get the arcs
    _, _, _, arcs_outer = compute_arcs(outer_tangents, outer_line_endpoints)
    _, _, _, arcs_opti = compute_arcs(outer_tangents, outer_line_endpoints)
    _, _, _, arcs_inner = compute_arcs(inner_tangents, inner_line_endpoints)

    # Get the midpoints of the arcs
    arc_midpoints_outer = find_arc_midpoints(arcs_outer)
    arc_midpoints_opti = find_arc_midpoints(arcs_opti)
    arc_midpoints_inner = find_arc_midpoints(arcs_inner)

    # Plot the midpoints of the outer and inner border
    plt.scatter([point[0] for point in arc_midpoints_outer], [point[1] for point in arc_midpoints_outer], color='red')
    plt.scatter([point[0] for point in arc_midpoints_inner], [point[1] for point in arc_midpoints_inner], color='blue')
    
    # Convert Sympy Floats in each tuple to standard Python floats
    arc_midpoints_outer = [[float(coord) for coord in point] for point in arc_midpoints_outer]
    arc_midpoints_opti = [[float(coord) for coord in point] for point in arc_midpoints_opti]
    arc_midpoints_inner = [[float(coord) for coord in point] for point in arc_midpoints_inner]

    opti_line_endpoints = outer_line_endpoints.copy()
    compteur = 0

    distances_turns = [[np.linalg.norm(np.array(opti_midpoint) - np.array(inner_midpoint)) / np.linalg.norm(np.array(outer_midpoint) - np.array(inner_midpoint))] for outer_midpoint, inner_midpoint, opti_midpoint in zip(arc_midpoints_outer, arc_midpoints_inner, arc_midpoints_opti)]
    while are_midpoints_above_threshold(arc_midpoints_opti, arc_midpoints_inner, arc_midpoints_outer, threshold) and compteur < step_max:
        for i in range(len(opti_line_endpoints) - 1):
            line_points = opti_line_endpoints[i]
            start_point = line_points[0]
            end_point = line_points[1]

            midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)

            dx = midpoint[0] - start_point[0]
            dy = midpoint[1] - start_point[1]

            opti_line_endpoints[i] = ((start_point[0] + dx*percentages[i], start_point[1] + dy*percentages[i]), (end_point[0] - dx*percentages[i+1], end_point[1] - dy*percentages[i+1]))

            # Get the same starting point for the first line
            if i == 0:
                opti_line_endpoints[i] = ((start_point[0], start_point[1]), (end_point[0] - dx*percentages[i+1], end_point[1] - dy*percentages[i+1]))
        
        for k in range(len(arc_midpoints_inner)):
            if not is_midpoint_above_threshold(arc_midpoints_opti[k], arc_midpoints_inner[k], arc_midpoints_outer[k], threshold, k):
                print("Turn n°", k, "has stopped increasing.")
                percentages[k+1] = 0  # Increase the percentage by a small amount  
        
        compteur += 1

        # Get the arcs
        _, _, _, arcs_opti = compute_arcs(outer_tangents, opti_line_endpoints)

        # Get the midpoints of the arcs
        arc_midpoints_opti = find_arc_midpoints(arcs_opti)

        # Plot the midpoint of the optimal path
        plt.scatter([point[0] for point in arc_midpoints_opti], [point[1] for point in arc_midpoints_opti], color='green')

        # Convert Sympy Floats in each tuple to standard Python floats
        arc_midpoints_opti = [[float(coord) for coord in point] for point in arc_midpoints_opti]

    return opti_line_endpoints, arc_midpoints_opti

def plot_correct_straight_lines(tangents, line_endpoints, arcs_end):
    # Generate a list of distinct colors
    distinct_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple', 'pink', 'lime', 'navy']
    color_cycler = cycle(distinct_colors)

    # Assign a unique color to each straight line for plotting
    line_colors = {i: next(color_cycler) for i in range(len(line_endpoints))}

    length_straight = []
    # Plot the modeled straight lines
    for i, line_points in enumerate(line_endpoints):
        
        # Plot the line from the start to the end of the straight section
        if i == 0:
            start_point = line_endpoints[i][0]
        else:
            # Ensure start_point is a tuple of standard Python floats
            start_point = (float(arcs_end[i-1][0]), float(arcs_end[i-1][1]))

        # Ensure end_point is a tuple of standard Python floats
        end_point = (float(line_endpoints[i][1][0]), float(line_endpoints[i][1][1]))
        
        line_length = np.sqrt((end_point[0]-start_point[0])**2 + (tangents[i][0]*end_point[0] + tangents[i][1] - (tangents[i][0]*start_point[0] + tangents[i][1]))**2)
        length_straight.append(line_length)
        plt.plot([start_point[0], end_point[0]], [tangents[i][0]*start_point[0] + tangents[i][1], tangents[i][0]*end_point[0] + tangents[i][1]], 
                    color=line_colors[i], linewidth=2, zorder=2)
        
    return length_straight

def save_dataframe_to_csv(tangents, length_straight, length_arc, radius, file_path):
    # Lists to store the DataFrame columns
    types = []
    section_lengths = []
    corner_radii = []

    # Loop through all sections and calculate the necessary information
    for i in range(len(tangents) - 1):
        
        # For straight lines
        types.append('Straight')
        section_lengths.append(length_straight[i])
        corner_radii.append(0)
        
        # For arcs
        # Determine if the arc is to the left or right by comparing the angle
        turn_direction = 'Left'
        types.append(turn_direction)
        section_lengths.append(abs(length_arc[i]))
        corner_radii.append(radius[i])
    
    types.append('Straight')
    section_lengths.append(length_straight[i+1])
    corner_radii.append(0)

    # Create the DataFrame
    track_df = pd.DataFrame({
        'Type': types,
        'Section Length': section_lengths,
        'Corner Radius': corner_radii
    })

    print(track_df)
    # Save the DataFrame to a CSV file with ';' as the separator
    track_df.to_csv(file_path, sep=';', index=False)

def plot_racetrack_oriented(start_point, sections, ax, color, label=None):
    current_x, current_y = start_point[0], start_point[1]
    current_angle = 0  # Starting angle is 0 degrees
    label_added = False  # Flag to check if label has been added

    for index, section in sections.iterrows():
        # Apply label only to the first element of each track
        element_label = label if not label_added else None
        if section['Type'] == 'Straight':
            # Calculate end point of the straight line
            dx = section['Section Length'] * np.cos(np.deg2rad(current_angle))
            dy = section['Section Length'] * np.sin(np.deg2rad(current_angle))
            ax.plot([current_x, current_x + dx], [current_y, current_y + dy], color=color)
            current_x += dx
            current_y += dy
        else:
             # Plot an arc
            radius = section['Corner Radius']
            length = section['Section Length']
            angle_change = np.rad2deg(length / radius)

            if section['Type'] == 'Left':
                arc_start_angle = current_angle - 90
                arc_end_angle = arc_start_angle + angle_change
                arc_center_x = current_x - radius * np.sin(np.deg2rad(current_angle))
                arc_center_y = current_y + radius * np.cos(np.deg2rad(current_angle))
            else: # Assuming 'Right' for any non-'Left' type
                arc_start_angle = current_angle - angle_change
                arc_end_angle = current_angle
                arc_center_x = current_x - radius * np.sin(np.deg2rad(arc_end_angle))
                arc_center_y = current_y + radius * np.cos(np.deg2rad(arc_end_angle))

            arc = patches.Arc((arc_center_x, arc_center_y), 2 * radius, 2 * radius, 
                              angle=0, theta1=min(arc_start_angle, arc_end_angle), 
                              theta2=max(arc_start_angle, arc_end_angle), color=color, label=element_label)
            ax.add_patch(arc)
            label_added = True

            current_angle = arc_end_angle 
            current_x = arc_center_x + radius * np.cos(np.deg2rad(current_angle))
            current_y = arc_center_y + radius * np.sin(np.deg2rad(current_angle))
            current_angle = current_angle + 90

    ax.set_title('Modelled Racetrack with Optimal Path')
    ax.set_xlabel('Meter')
    ax.set_ylabel('Meter')
    ax.axis('equal')

