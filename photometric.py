import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import lsqr

def normals2surface(x_normals, y_normals, lambda1, lambda2):
    """
    Given x normals, y normals, and regularization parameters,
    return surface approximation fxy. Based on regularized least squared,
    using scipy.linalg.sparse.lsqr.
    """
    rows, cols = x_normals.shape
    Dx = sps.diags([-np.ones(rows), np.ones(rows)], [0,1], shape=(rows,rows))
    A1 = sps.kron(sps.eye(cols), Dx)
    Dy = sps.diags([-np.ones(cols), np.ones(cols)], [0,1], shape=(cols,cols))
    A2 = sps.kron(Dy, sps.eye(rows))
    A = sps.vstack((A1, A2, \
               np.sqrt(lambda1)*sps.eye(A1.shape[0]), \
               np.sqrt(lambda1)*sps.eye(A2.shape[0]), \
               np.sqrt(lambda2)*A1, \
               np.sqrt(lambda2)*A2))
    surface_normals = np.vstack((x_normals.reshape((-1,1), order='F'), y_normals.reshape((-1,1), order='F')))
    b = np.vstack((surface_normals, np.zeros((2*A1.shape[0] + 2*A2.shape[0],1))))
    fxy = lsqr(A, b, atol=1e-6, btol=1e-6, iter_lim=5000)
    fxy = fxy[0].reshape(x_normals.shape, order='F')
    return fxy

def frankotchellappa(dzdx, dzdy):
    rows, cols = dzdx.shape

    wx, wy = np.meshgrid((range(cols)-(np.fix(cols/2)+1))/(cols-np.mod(cols,2)), \
                         (range(rows)-(np.fix(rows/2)+1))/(rows-np.mod(rows,2)))
    wx = np.fft.ifftshift(wx)
    wy = np.fft.ifftshift(wy)

    DZDX = np.fft.fft2(dzdx)
    DZDY = np.fft.fft2(dzdy)

    Z = 0-1j*(np.multiply(wx, DZDX) + np.multiply(wy, DZDY))/(np.square(wx) + np.square(wy) + np.finfo(float).eps)

    z = np.real(np.fft.ifft2(Z))
    z -= z.min()
    return z/2

def hw8pw(I, L):
    """
    Compute unit normals from image and lighting information.

    Inputs:       I is an (m x n x d) matrix whose d slices contain m x n
                  double-precision images of a common scene under different
                  lighting conditions
                  L is a (3 x d) matrix such that L(:,i) is the lighting
                  direction vector for image I(:,:,i)

    Outputs:      N is an (m x n x 3) matrix containing the unit-norm surface
                  normal vectors for each pixel in the scene
    """
    m,n,d = I.shape
    I = I.reshape((m*n,d), order='F')
    g = np.dot(I, np.linalg.pinv(L)).reshape((m,n,3), order='F')

    N = g/(np.linalg.norm(g, axis=2)[:,:,np.newaxis])
    return N

def D(n):
    """
    Returns n-by-n circulant first difference matrix.
    """
    Dn = sps.diags([-np.ones(n), np.ones(n)], [0,1], shape=(n,n)).tolil()
    Dn[-1,0] = 1
    return Dn

def hw8p9a(DFDX, DFDY, regparam):
    """
    Inputs:       DFDX and DFDY are m x n matrices containing the partial
                  derivatives of f with respect to x and y at each pixel
                  coordinate

                  lambda >= 0 is the Tikhonov regularization parameter

    Outputs:      A is a 3mn x mn sparse matrix and b is a 3mn x 1 vector
                  such that solving

                  fxy = argmin_f ||b - Af||^2

                  yields the surface FXY = reshape(fxy, [m, n]) with partial
                  derivatives DFDX and DFDY

    Note:         Uses CIRCULANT first difference matrices
    """
    m,n = DFDX.shape

    A = sps.vstack((sps.kron(D(n), sps.eye(m)),
              sps.kron(sps.eye(n), D(m))))
    b = np.vstack((DFDX.reshape((-1,1), order='F'), DFDY.reshape((-1,1), order='F')))

    # add regularization
    A = sps.vstack((A, np.sqrt(regparam)*sps.eye(m*n)))
    b = np.vstack((b, np.zeros((m*n, 1))))

    return A, b
