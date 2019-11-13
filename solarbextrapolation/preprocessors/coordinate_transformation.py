import numpy as np


def cart2sph3d(data, origin=[0, 0, 0], zenith=[0, 0, 1], azimuth=[1, 0, 0]):
    """
    This function converts 3d data in Cartesian coordinate system to spherical 
    coordinate system.

    The points are vectors v_1,v_2, …, v_n, and the vectors are composed of x, 
    y and z coordinates, so that v_1=(x_1, y_1, z_1), then your points matrix P 
    would look like:
    P = [[x_1, x_2, …, x_n], 
         [y_1, y_2, …, y_n],
         [z_1, z_2, …, z_n]]

    Parameters
    ----------
    data: list or 'numpy.ndarray'
          A list or numpy.ndarray object of vectors that is going to be 
          converted from Cartesian coordinate system to spherical coordinate 
          system. Currently, the shape must be (3, n).

    origin: list or 'numpy.ndarray'
            The position of origin of spherical coordinate system in Cartesian 
            coordinate system. The default value is [0, 0, 0] (origin point of 
            Cartesian coordinate system).

    zenith: list or 'numpy.ndarray'
            The vector to represent the direction of zenith of spherical 
            coordinate system in Cartesian coordinate system. The default value
            is [0, 0, 1] (z direction of Cartesian coordinate system).

    azimuth: list or 'numpy.ndarray'
             The vector to represent the direction of zenith of spherical 
             coordinate system in Cartesian coordinate system. The default value
             is [1, 0, 0] (x direction of Cartesian coordinate system).
    """
    origin  =  np.array(origin ).reshape(3, -1).astype(float)
    zenith  =  np.array(zenith ).reshape(-1, 3).astype(float)
    azimuth =  np.array(azimuth).reshape(-1, 3).astype(float)
    zenith  *= 1. / np.sqrt(zenith @ zenith.T)
    azimuth *= 1. / np.sqrt(azimuth @ azimuth.T)
    assert 1e-10 > (zenith @ azimuth.T), \
        "Zenith is not perpendicular to azimuth!"
    y_0     =  np.cross(zenith, azimuth)
    y_0     *= 1. / np.sqrt(y_0 @ y_0.T) 

    # Currently data only support shape (3, n), an elegant way to increase its 
    # capability to support different shapes should be added
    data = np.array(data).astype(float)
    assert (1 < len(data.shape)) and (3 == data.shape[0]), \
        "Shape of vector matrix is incorrect! It should be (3, n), where n is the number of vectors"
    data -= origin

    r     = np.sqrt(((data)**2).sum(axis=0)).reshape(1, -1)
    z     = zenith @ data
    theta = np.zeros(r.shape)
    theta[r != 0] = np.arccos(z[r != 0] / r[r != 0])

    xy         = data - zenith @ data * zenith.T
    x          = azimuth @ data
    y          = y_0 @ (xy - x * azimuth.T)
    phi        = np.zeros(theta.shape)
    phi[(0 < x) & (0 != y)] = np.arctan(y[(0 < x) & (0 != y)] / x[(0 < x) & (0 != y)])
    phi[(0 > x) & (0 != y)] = np.arctan(y[(0 > x) & (0 != y)] / x[(0 > x) & (0 != y)]) + np.pi * np.sign(y[(0 > x) & (0 != y)])
    phi[(0 == x) & (y != 0)] += 1 / 2 * np.pi * np.sign(y[(0 == x) & (y != 0)])
    phi[(0 > x) & (y == 0)] += np.pi

    r     = r.reshape(-1)
    theta = theta.reshape(-1)
    phi   = phi.reshape(-1)
    return np.array([r, theta, phi]).reshape(3, -1)


def sph2cart(data):
    """
    Auxiliary function which should be set as a private method later

    Parameters
    ----------
    Parameters
    ----------
    data: list or 'numpy.ndarray'
          A list or numpy.ndarray object of vectors that is going to be 
          converted from spherical coordinate system to Cartesian coordinate 
          system. Currently, the shape must be (3, n).
    """
    data  = np.array(data)
    r     = data[0, :]
    theta = data[1, :]
    phi   = data[2, :]
    x     = r * np.sin(theta) * np.cos(phi)
    y     = r * np.sin(theta) * np.sin(phi)
    z     = r * np.cos(theta)
    return np.array([x, y, z]).reshape(3, -1)


def sph2cart3d(data, X_0=[1, np.pi / 2, 0], Y_0=[1, np.pi / 2, np.pi / 2], Z_0=[1, 0, 0], Origin=[0, 0, 0]):
    """
    This function converts 3d data in Cartesian coordinate system to spherical 
    coordinate system.

    The points are vectors v_1,v_2, …, v_n, and the vectors are composed of x, 
    y and z coordinates, so that v_1=(r_1, theta_1, phi_1), then your points 
    matrix P would look like:
    P = [[r_1,     r_2,     …, r_n], 
         [theta_1, theta_2, …, theta_n],
         [phi_1,   phi_2,   …, phi_n]]

    Note that X_0, Y_0, Z_0 are three vectors rather x, y, z, which are x, y, z 
    components of all vectors.

    Parameters
    ----------
    data: list or 'numpy.ndarray'
          A list or numpy.ndarray object of vectors that is going to be converted
          from Cartesian coordinate system to spherical coordinate system. 
          Currently, the shape must be (3, n).

    X_0: list or 'numpy.ndarray'
         The vector to represent the direction of axis of Cartesian coordinate 
         system in spherical coordinate system. The default value is 
         [1, pi/2, 0] (azimuth direction of spherical coordinate system)

    Y_0: list or 'numpy.ndarray'
         The vector to represent the direction of axis of Cartesian coordinate 
         system in spherical coordinate system. The default value is 
         [1, pi/2, pi/2] (normal vector of x-z plane)

    Z_0: list or 'numpy.ndarray'
         The vector to represent the direction of axis of Cartesian coordinate 
         system in spherical coordinate system. The default value is [1, 0, 0] 
         (zenith direction of spherical coordinate system)

    Origin: list or 'numpy.ndarray'
            The position of origin of Cartesian coordinate system in spherical
            coordinate system. The default value is [0, 0, 0] (origin point of 
            spherical coordinate system)
    """
    X_0  = np.array(X_0).reshape(3, -1).astype(float)
    Y_0  = np.array(Y_0).reshape(3, -1).astype(float)
    Z_0  = np.array(Z_0).reshape(3, -1).astype(float)
    X_0 *= 1. / X_0[0]
    Y_0 *= 1. / Y_0[0]
    Z_0 *= 1. / Z_0[0]
    M    = sph2cart(np.array([X_0, Y_0, Z_0]).reshape(3, 3).T).T
    assert (1e-10 > M[0, :] @ M[1, :]) and (1e-10 > M[1, :] @ M[2, :]) and ((1e-10 > M[2, :] @ M[0, :])), \
        "Normal vectors of Cartesian coordinate system are not orthogonal!"

    # Currently data only support shape (3, n), an elegant way to increase its 
    # capability to support different shapes should be added
    data = np.array(data).astype(float)
    assert (1 < len(data.shape)) and (3 == data.shape[0]), \
        "Shape of vector matrix is incorrect! It should be (3, n), where n is the number of vectors"

    Origin = np.array(Origin).reshape(3, -1).astype(float)
    Origin = sph2cart(Origin)

    data  = M @ sph2cart(data) - Origin
    return data