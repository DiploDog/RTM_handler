from utils.profile_processor import PreStartPoint


def dms(coord, course):
    """
    The coordinate format reducer.

    Parameters
    ----------
    coord : pd.DataFrame, pd.Series, np.array
        A pair of coordinates
    course : str
        The direction towards which a railway car is moving:
        'N' for North
        'S' for South
        'E' for East
        'W' for West

    Returns
    -------
    degrees: float
        The degree part of the decimal coordinate
    minutes: float
        The minutes part of the decimal coordinate
    sign: int
        The sign according to the travel direction
    """

    direction = {'N': -1, 'S': 1, 'E': -1, 'W': 1}
    coord = coord.astype('float64')
    degrees = coord.astype('int64') // 100
    minutes = coord - 100 * degrees
    sign = course.map(direction)

    return degrees, minutes, sign


def decimal_degrees(degrees, minutes, sign):
    """
    The decimal format of degree transformer.

    Parameters
    ----------
    degrees: float
        The degree part of the decimal coordinate
    minutes: float
        The minutes part of the decimal coordinate
    sign: int
        The sign according to the travel direction

    Returns
    -------
    float:
        Coordinate in decimal format
    """
    return round((degrees + minutes/60) * sign, 6)

