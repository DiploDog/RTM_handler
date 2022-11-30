from utils.profile_processor import PreStartPoint

# class CoordsHandler:
#
#     def __init__(self):
#         pass

def dms( coord, course):
    direction = {'N': -1, 'S': 1, 'E': -1, 'W': 1}
    coord = coord.astype('float64')
    degrees = coord.astype('int64') // 100
    minutes = coord - 100 * degrees
    sign = course.map(direction)

    return degrees, minutes, sign

def decimal_degrees( degrees, minutes, sign):
    return round((degrees + minutes/60) * sign, 6)

