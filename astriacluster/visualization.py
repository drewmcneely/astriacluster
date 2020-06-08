human_height = 1.6

def aframe_point_component_position(data):
    max_r = 1e8

    data = data[['r_0', 'r_1', 'r_2']]
    data = data / max_r * human_height # rescale data to roughly the average human height in meters
    data['r_2'] = data['r_2'] + human_height # Move origin up to human height
    return(data)

def aframe_point_component_angularmomentum(data):
    max_h = 1e11
    min_h_z = -5e10

    data = data[['h_0', 'h_1', 'h_2']]
    data = data / max_h * human_height # rescale data to roughly the average human height in meters
    data['h_2'] = data['h_2'] + (-min_h_z / max_h * human_height) # Move origin so that min_h_z sits on floor
    return(data)
