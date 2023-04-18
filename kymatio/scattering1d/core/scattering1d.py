
def scattering1d(U_0, backend, filters, log2_stride, average_local):
    """
    Main function implementing the 1-D scattering transform.

    Parameters
    ----------
    U_0 : Tensor
        an backend-compatible array of size `(B, 1, N)` where `B` is batch size
        and `N` is the padded signal length.
    backend : module
        Kymatio module which matches the type of U_0.
    psi1 : list
        a list of dictionaries, expressing wavelet band-pass filters in the
        Fourier domain for the first layer of the scattering transform.
        Each `psi1[n1]` is a dictionary with keys:
            * `j`: int, subsampling factor
            * `xi`: float, center frequency
            * `sigma`: float, bandwidth
            * `levels`: list, values taken by the wavelet in the Fourier domain
                        at different levels of detail.
        Each psi1[n]['levels'][level] is an array with size N/2**level.
    psi2 : dictionary
        Same as psi1, but for the second layer of the scattering transform.
    phi : dictionary
        a dictionary expressing the low-pass filter in the Fourier domain.
        Keys:
        * `j`: int, subsampling factor (also known as log_T)
        * `xi`: float, center frequency (=0 by convention)
        * `sigma`: float, bandwidth
        * 'levels': list, values taken by the lowpass in the Fourier domain
                    at different levels of detail.
    log2_stride : int >=0
        Yields coefficients with a temporal stride equal to 2**log2_stride.
    average_local : boolean
        whether to locally average the result by means of a low-pass filter phi.
    """

    # compute the Fourier transform
    U_0_hat = backend.rfft(U_0)

    # Get S0
    phi = filters[0]
    log2_T = phi['j']
    stride = 2**log2_stride

    # First order:
    psi1 = filters[1]
    # Convolution + downsampling
    j1 = psi1[0]['j']
    k1 = min(j1, log2_stride) if average_local else j1
    # convolution for signal + filterbank
    U_1_c = backend.cdgmm(U_0_hat, psi1[0]['levels'][0])
    #U_1_hat = backend.subsample_fourier(U_1_c, 2**k1)
    #U_1_c = backend.ifft(U_1_hat)
    # inverse fft
    U_1_c = backend.ifft(U_1_c)
    return U_1_c


__all__ = ['scattering1d']
