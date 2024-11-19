import numpy as np
import matplotlib.pyplot as plt
import os

# Define the rotation matrix about the x-axis
def rotation_matrix_x(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def line_plane_intersection(p1, p2, normal_vector, d):
    """
    Find the intersection of the line through p1 and p2 with the plane defined by normal_vector and d.
    """
    line_dir = p2 - p1
    t = -(np.dot(normal_vector, p1) + d) / np.dot(normal_vector, line_dir)
    intersection = p1 + t * line_dir
    return intersection

def generate(grid_x, grid_y, inplane_offset, outplane_offset, image_height, plot = False):
    inplane_offset = np.round(inplane_offset, 3)
    outplane_offset = np.round(outplane_offset, 3)
    
    os.makedirs(f'projections', exist_ok=True)
    path = f'projections/SkewedProjection_ResOf{grid_x}x{grid_y}_OffsetOf{inplane_offset}x{outplane_offset}_ImgHeight{image_height}.npz'
    if os.path.exists(path):
        squares = np.load(path)['squares']
        return squares

    scale_x = grid_x / grid_x
    scale_y = grid_y / grid_x

    # Find projection height
    horz_offset = inplane_offset/image_height
    vert_offset = outplane_offset/image_height
    closest_offset = np.sqrt(horz_offset**2 + vert_offset**2)
    farthest_offset = np.sqrt((1+horz_offset)**2 + vert_offset**2)
    offset_angle = np.atan(vert_offset/horz_offset)
    proj_height = (np.sin(np.pi - offset_angle)/(np.cos(np.pi - offset_angle) + farthest_offset - closest_offset))*scale_y/2

    # Generate grid points
    x = np.linspace(0, scale_x, grid_x)
    y = np.linspace(0, scale_y, grid_y)
    x, y = np.meshgrid(x, y)

    x_flat = x.flatten()
    y_flat = y.flatten()

    if plot:
        # Create a list of points
        display_grid = [[x.flatten()[i], y.flatten()[i], np.zeros_like(x.flatten())[i]] for i in range(len(x.flatten()))]

    # Define the key points
    center_point = np.array([0.5 * scale_x, 0.5 * scale_y, proj_height])
    point_a = np.array([0.5 * scale_x, scale_y, 0])
    point_b = np.array([0.5 * scale_x, 0, 0])

    # Define lines
    line1 = center_point - point_a
    line2 = point_b - point_a

    # Compute angles between lines
    theta1 = np.arccos(np.dot(line2, line1) / (np.linalg.norm(line1) * np.linalg.norm(line2)))

    # Calculate theta3 and theta4
    theta2 = np.pi - (theta1 + offset_angle)
    theta3 = np.pi - (theta1 + theta2)

    # Use the law of sines to find distance D
    D = scale_y * np.sin(theta2) / np.sin(theta3)

    # Calculate the new point along line1 at distance D from point_a
    direction = line1 / np.linalg.norm(line1)

    # Define the plane using three points
    point1 = np.array([0, 0, 0])
    point2 = np.array([scale_x, 0, 0])
    point3 = point_a + D * direction

    # Create the plane
    v1 = point2 - point1
    v2 = point3 - point1
    normal_vector = np.cross(v1, v2)
    d = -point1.dot(normal_vector)

    # Calculate the angle between the drawn plane and the x-y plane
    angle_with_xy = -np.arctan2(np.linalg.norm(np.cross(normal_vector, [0, 0, 1])), np.dot(normal_vector, [0, 0, 1]))

    # Rotate points
    rotation_matrix = rotation_matrix_x(angle_with_xy)

    squares = []
    virtual_grid = []
    for i in range(np.size(x)):
        if i % 1000000 == 999999:
            print(f"Generating Points {i + 1}/{grid_y*grid_x}...")
        virtual_grid.append(np.dot(rotation_matrix,
                                   line_plane_intersection([x_flat[i], y_flat[i], 0],
                                                            np.array(center_point),
                                                            normal_vector, d)))

    x_skewed = np.array([p[0] for p in virtual_grid]).reshape(x.shape)
    y_skewed = np.array([p[1] for p in virtual_grid]).reshape(y.shape)

    if plot:
        center_point = np.dot(rotation_matrix, center_point)
        display_grid = [np.dot(rotation_matrix, p) for p in display_grid]
        v_points_x = [p[0] for p in display_grid]
        v_points_y = [p[1] for p in display_grid]
        v_points_z = [p[2] for p in display_grid]

    for i in range(grid_y - 1):
        for j in range(grid_x - 1):
            if ((grid_x-1)*i + j) % 1000000 == 999999:
                print(f"Drawing Pixel {(grid_x-1)*i + j + 1}/{(grid_y-1)*(grid_x-1)}...")

            # Original square corners
            square = [
                (x_skewed[i, j], y_skewed[i, j]),      # top_left
                (x_skewed[i, j + 1], y_skewed[i, j + 1]),  # top_right
                (x_skewed[i + 1, j + 1], y_skewed[i + 1, j + 1]),  # bottom_right
                (x_skewed[i + 1, j], y_skewed[i + 1, j])   # bottom_left
            ]
            
            squares.append(np.array(square))

            if plot:
                # Plotting the points for the current square
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot the grid points
                ax.scatter(v_points_x, v_points_y, v_points_z, c='b', marker='o', alpha=0.2)

                # Plot the points of the current square
                square_x = [p[0] for p in square]
                square_y = [p[1] for p in square]
                square_z = np.zeros_like(square_x)
                ax.scatter(square_x, square_y, square_z, c='g', marker='o', s=100)

                # Plot the center point
                ax.scatter(center_point[0], center_point[1], center_point[2], c='r', marker='o', s=100)

                # Coordinates of the farthest corner points
                corners = [
                    (0, 0, 0),                # Bottom-left corner
                    (scale_x, 0, 0),          # Bottom-right corner
                    (scale_x, scale_y, 0),    # Top-right corner
                    (0, scale_y, 0)           # Top-left corner
                ]
                corners = [np.dot(rotation_matrix, p) for p in corners]

                # Plot thin red lines between the center point and the farthest corner points
                for corner in corners:
                    ax.plot(
                        [center_point[0], corner[0]], 
                        [center_point[1], corner[1]], 
                        [center_point[2], corner[2]], 
                        'r-', 
                        linewidth=0.5
                    )

                from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                # Define the intersection points on the x-y plane
                intersection_points = []

                for corner in corners:
                    intersection_point = line_plane_intersection(center_point, np.array(corner), np.array([0, 0, 1]), 0)
                    intersection_points.append(intersection_point)

                # Extract coordinates for the trapezoid
                trapezoid_x = [p[0] for p in intersection_points]
                trapezoid_y = [p[1] for p in intersection_points]
                trapezoid_z = [p[2] for p in intersection_points]

                # Plot the trapezoid
                verts = [list(zip(trapezoid_x, trapezoid_y, trapezoid_z))]
                ax.add_collection3d(Poly3DCollection(verts, color='green', alpha=0.3))

                # Set labels
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                ax.set_xlim(0, scale_x)
                ax.set_ylim((scale_y - scale_x)/2, (scale_y + scale_x)/2)
                ax.set_zlim(-0.5, 0.5)
                ax.set_box_aspect([1, 1, 1])

                ax.view_init(elev=15, azim=30)
                # Show the plot
                plt.show()

    if plot:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        print("Skewed pixels:")
        for idx, square in enumerate(squares):
            print(f"Pixel {idx + 1}: {square[0]}, {square[1]}, {square[2]}, {square[3]}")
    
    np.savez_compressed(path, squares=squares)
    squares = np.load(path)['squares']
    return squares