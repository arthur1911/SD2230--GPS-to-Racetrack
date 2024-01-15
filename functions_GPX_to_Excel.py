import gpxpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from itertools import cycle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sympy import symbols, Eq, solve, atan2, pi

def get_gpx(file_path):
    # Read GPX file
    with    open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    latitudes = []
    longitudes = []

    # Extract coordinates from GPX file
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                latitudes.append(point.latitude)
                longitudes.append(point.longitude)
    return latitudes, longitudes


# Function to convert lat/lon to meters
def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    # Constants for the conversions
    m_per_deg_lat = 111000
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(ref_lat))
    
    # Conversion
    x = (lon - ref_lon) * m_per_deg_lon
    y = (lat - ref_lat) * m_per_deg_lat
    
    return (x, y)

def plot_route(points, condition, c='b'):
    # Plotting the route
    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b')  # s is the size of the point
    plt.title('GPX Track')
    if condition == 'coordinates':
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    else:
        plt.xlabel('Meter')
        plt.ylabel('Meter')
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def plot_colored_route(point_colors, smoothed_points):
    plt.figure(figsize=(10, 6))
    # Plot points in bulk, grouped by color
    for color in set(point_colors):
        # Select points that match the current color
        mask = point_colors == color
        plt.scatter(smoothed_points[mask, 0], smoothed_points[mask, 1], color=color, s=10)

    plt.title('GPX Track with Distinctly Colored Straight Sections')
    plt.xlabel('Meter')
    plt.ylabel('Meter')
    plt.axis("equal")
    plt.grid(True)
    plt.show()

def moving_average_filter(coordinates, window_size):
    # Convert to a pandas DataFrame for convenience
    df = pd.DataFrame(coordinates, columns=['longitude', 'latitude'])
    
    # Apply the moving average filter and return as numpy array
    df_smooth = df.rolling(window=window_size, center=True, min_periods=1).mean()
    return df_smooth.to_numpy()

def spot_straight_lines(points, min_points_straight, error_threshold):
    straight_lines = []
    current_line_points = []
    line_indices = [-1] * len(points)  # Initialize with -1 indicating no straight line
    scaler = StandardScaler()  # Create a standard scaler instance

    for i, point in enumerate(points):
        current_line_points.append(point)

        # If we have enough points, check if they form a straight line
        if len(current_line_points) >= min_points_straight:
            # Standardize the current line points
            current_line_points_array = np.array(current_line_points)
            standardized_points = scaler.fit_transform(current_line_points_array)

            model = LinearRegression()
            # Use the standardized coordinates for fitting
            X = standardized_points[:, 0].reshape(-1, 1)
            y = standardized_points[:, 1]
            model.fit(X, y)

            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)

            if mse <= error_threshold:
                # Extend the current line if the error is within the threshold
                if line_indices[i - min_points_straight] != -1:
                    # Continue the previous line
                    line_index = line_indices[i - min_points_straight]
                else:
                    # Start a new line
                    line_index = len(straight_lines)
                    straight_lines.append([])

                # Add points to the straight line
                straight_lines[line_index].extend(current_line_points)
                line_indices[i - min_points_straight + 1:i + 1] = [line_index] * min_points_straight
                current_line_points = []

            elif len(current_line_points) > min_points_straight:
                # If error exceeds with additional points, reset the line_points
                current_line_points = current_line_points[1:]

    return straight_lines, line_indices

def get_tagents_and_endpoints(straight_lines):
    tangents = []
    line_endpoints = []  

    # Plot the modeled straight lines
    for i, line_points in enumerate(straight_lines):
        if len(line_points) > 1:
            # Fit the linear model to the straight line points
            model = LinearRegression()
            X = np.array([p[0] for p in line_points]).reshape(-1, 1)
            y = np.array([p[1] for p in line_points])
            model.fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            tangents.append((slope, intercept))

            # Capture start and end points
            start_point = line_points[0]
            end_point = line_points[-1]

            # Get the abscissa values for the start and end points
            start_point_x = start_point[0]
            end_point_x = end_point[0]

            # Get the ordinates values for the start and end points
            start_point_y = slope * start_point_x + intercept
            end_point_y = slope * end_point_x + intercept

            line_endpoints.append(((start_point_x, start_point_y), (end_point_x, end_point_y)))  # Store as a tuple
    return tangents, line_endpoints

def plot_straight_lines(tangents, line_endpoints, line_colors):
    # Plot the modeled straight lines
    for i, points in enumerate(line_endpoints):
            # Plot the line from the start to the end of the straight section
            start_point_x, start_point_y = points[0]
            end_point_x, end_point_y = points[1]
            
            plt.plot([start_point_x, end_point_x], [start_point_y, end_point_y], 
                    color=line_colors[i], linewidth=2, zorder=2)
            
    plt.title('GPX Track with Modeled Straight Lines')
    plt.xlabel('Meter')
    plt.ylabel('Meter')
    plt.axis("equal")
    plt.grid(True)
    plt.show()


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
    perpendicular_slope = -1/m1
    perpendicular_eq = Eq(y - A[1], perpendicular_slope * (x - A[0]))
    
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

def plot_corrected_straigth_lines(line_endpoints, arcs_end, tangents, line_colors):
    length_straight = []
    
    # Plot the modeled straight lines
    for i, line_points in enumerate(line_endpoints):
        
        # Plot the line from the start to the end of the straight section
        if i == 0:
            start_point = line_points[0]
        else:
            # Ensure start_point is a tuple of standard Python floats
            start_point = (float(arcs_end[i-1][0]), float(arcs_end[i-1][1]))

        # Ensure end_point is a tuple of standard Python floats
        end_point = line_points[1]
        
        line_length = np.sqrt((end_point[0]-start_point[0])**2 + (tangents[i][0]*end_point[0] + tangents[i][1] - (tangents[i][0]*start_point[0] + tangents[i][1]))**2)
        length_straight.append(line_length)
        plt.plot([start_point[0], end_point[0]], [tangents[i][0]*start_point[0] + tangents[i][1], tangents[i][0]*end_point[0] + tangents[i][1]], 
                    color=line_colors[i], linewidth=2, zorder=2)

    plt.title('Corrected Track with Modeled Straight Lines and Arcs')
    plt.xlabel('Meter')
    plt.ylabel('Meter')
    plt.axis("equal")
    plt.grid(True)
    plt.show()

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