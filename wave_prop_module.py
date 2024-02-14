import numpy as np


class Wave2d:
    """
    Given a wave components on a plane and optical parameters,
    calculates the wave components at another plane.
    """

    def __init__(self, numPx: list = [1392, 1040], 
                 sizePx: list = [0.00645, 0.00645], wl: float = 658*1e-6):
        
        ## setup params
        # wavelength, camera specs, resolutions, and limits

        self.mm = 1e-3 # standard otherwise specified
        ## Inputs
        self.wl = wl*self.mm

        ## Camera-based calculations or planes size and resolution
        self.numPx = numPx # Pixels along the [x-axis, y-axis] or samples
        self.sizePx = [sizePx[0]*self.mm, sizePx[1]*self.mm] # pixel size [x-axis, y-axis]

        self.sizeSensorX = self.numPx[0]*self.sizePx[0] # sensor size along x-axis
        self.sizeSensorY = self.numPx[1]*self.sizePx[1] # sensor size along y-axis

        # Sampling plane 2x for linearization
        self.Sx = 2*self.sizeSensorX
        self.Sy = 2*self.sizeSensorY

        self.samplingRate = 1/self.sizePx[0]

        # max freq to prevent aliasing by the transfer function
        self.delU = 1/self.Sx
        self.delV = 1/self.Sy

        self.freqRows = np.linspace(-1/(2*self.sizePx[0]), 1/(2*self.sizePx[0]), int(1/(self.sizePx[0]*self.delU))) ## -maxFreq/2, +maxFreq/2
        self.freqCols = np.linspace(-1/(2*self.sizePx[1]), 1/(2*self.sizePx[1]), int(1/(self.sizePx[1]*self.delV)))

        self.u, self.v = np.meshgrid(self.freqRows, self.freqCols)

        self.wavefield_z0 = None
        self.wavefield_z1 = None

        self.fft_wave_z0 = None
        self.fft_wave_z1 = None
        self.z = None
    
    def propogate(self, dist: float):
        assert self.wavefield_z0 is not None, "Use method wavefied first"
        assert self.fft_wave_z0 is not None, "Use method wavefied first"

        self.z = dist*self.mm # distance to propogate along the z axis
        self.uLimit = 1/(np.sqrt((2*self.delU*self.z)**2 + 1)* self.wl)
        self.vLimit = 1/(np.sqrt((2*self.delV*self.z)**2 + 1)* self.wl)
        
        k = self.wl**(-2) - self.u**2 - self.v**2

        # removing evanscent waves: limiting transfer function does that but to remove 
        # numerical errors that may occur, safer to zero freqs > 1/wl
        if not np.all(k > 0):
            k[k < 0] = 0

        w = np.sqrt(k)
        H = np.exp(1j*2*np.pi*w*self.z)

        # limiting frequencies above uLimit and vLimit of the transfer function
        mask = np.ones_like(H)
        mask[np.logical_or(np.abs(self.u) > int(self.uLimit), np.abs(self.v) >= int(self.vLimit))] = 0

        H = mask*H
        self.fft_wave_z1 = H*self.fft_wave_z0
        self.wavefield_z1 = np.fft.ifft2(np.fft.fftshift(self.fft_wave_z1))
        self.wavefield_z1 = self.wavefield_z1[
            int(self.fft_wave_z1.shape[0]/2 - self.wavefield_z0.shape[0]/2):int(self.fft_wave_z0.shape[0]/2 + self.wavefield_z0.shape[0]/2),
            int(self.fft_wave_z1.shape[1]/2 - self.wavefield_z0.shape[1]/2):int(self.fft_wave_z0.shape[1]/2 + self.wavefield_z0.shape[1]/2)    
            ]
        
        return self.wavefield_z1

    def wavefield(self, wave: np.array(np.complex128)):
        assert [wave.shape[1], wave.shape[0]] == self.numPx, "Incorrect number of pixels specified in constructor"
        # not using this wavefield to calculate here for speed as function may be used repeatedly

        linImg = np.zeros([int(self.Sy/self.sizePx[1]), int(self.Sx/self.sizePx[0])], dtype=np.complex128)
        linImg[int(linImg.shape[0]/2 - wave.shape[0]/2):int(linImg.shape[0]/2 + wave.shape[0]/2), 
            int(linImg.shape[1]/2 - wave.shape[1]/2):int(linImg.shape[1]/2 + wave.shape[1]/2)] = wave
 
        self.wavefield_z0 = wave
        self.fft_wave_z0 = np.fft.fftshift(np.fft.fft2(linImg))

    def setup_limit_info(self):
        assert self.z != None, "Distance is set to none"
        ## Nice to know 1: max freq and angle that the camera/plane can record without aliasing
        maxFreqPossible = self.samplingRate/2
        maxAnglePossible = self.samplingRate*self.wl # cosine

        ## Nice to know 2: max freq and angle that the setup allows. Freqs above these will not
        ## reach the next plane
        maxAngleSetupX = self.sizeSensorX/(2*np.linalg.norm([self.z, self.sizeSensorX])) # cosine \thetaX
        maxAngleSetupY = self.sizeSensorY/(2*np.linalg.norm([self.z, self.sizeSensorY])) # cosing \thetaY

        maxFreqSetupX = maxAngleSetupX/self.wl
        maxFreqSetupY = maxAngleSetupY/self.wl

        print(f'Max Freq and Angle the camera/plane can record without aliasing: {maxFreqPossible*self.mm} cycles/mm | {maxAnglePossible} radians')
        
        print(f'Max Freq and Angle the setup allows (freqs > do not reach the next plane): {(maxFreqSetupX*self.mm, maxFreqSetupY*self.mm)} cycles/mm | {(maxAngleSetupX, maxAngleSetupY)} radians')

