import socket
import struct
import numpy
import time

class _BaseTrignoDaq(object):
    """
    Delsys Trigno wireless EMG system.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    host : str
        IP address the TCU server is running on.
    cmd_port : int
        Port of TCU command messages.
    data_port : int
        Port of TCU data access.
    rate : int
        Sampling rate of the data source.
    total_channels : int
        Total number of channels supported by the device.
    timeout : float
        Number of seconds before socket returns a timeout exception

    Attributes
    ----------
    BYTES_PER_CHANNEL : int
        Number of bytes per sample per channel. EMG and accelerometer data
    CMD_TERM : str
        Command string termination.

    Notes
    -----
    Implementation details can be found in the Delsys SDK reference:
    http://www.delsys.com/integration/sdk/
    """

    BYTES_PER_CHANNEL = 4
    CMD_TERM = '\r\n\r\n'

    def __init__(self, host, cmd_port, data_port, total_channels, timeout):
        self.host = host
        self.cmd_port = cmd_port
        self.data_port = data_port
        self.total_channels = total_channels
        self.timeout = timeout

        self._min_recv_size = self.total_channels * self.BYTES_PER_CHANNEL

        self._initialize()

    def _initialize(self):

        # create command socket and consume the servers initial response
        self._comm_socket = socket.create_connection(
            (self.host, self.cmd_port), self.timeout)
        self._comm_socket.recv(1024)

        # create the data socket
        self._data_socket = socket.create_connection(
            (self.host, self.data_port), self.timeout)

    def start(self):
        """
        Tell the device to begin streaming data.

        You should call ``read()`` soon after this, though the device typically
        takes about two seconds to send back the first batch of data.
        """
        self._send_cmd('START')

    def read(self, num_samples):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Parameters
        ----------
        num_samples : int
            Number of samples to read per channel.

        Returns
        -------
        data : ndarray, shape=(total_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        l_des = num_samples * self._min_recv_size
        l = 0
        packet = bytes()
        while l < l_des:
            try:
                packet += self._data_socket.recv(l_des - l)
            except socket.timeout:
                l = len(packet)
                packet += b'\x00' * (l_des - l)
                raise IOError("Device disconnected.")
            l = len(packet)

        data = numpy.asarray(
            struct.unpack('<'+'f'*self.total_channels*num_samples, packet))
        data = numpy.transpose(data.reshape((-1, self.total_channels)))

        return data

    def stop(self):
        """Tell the device to stop streaming data."""
        self._send_cmd('STOP')

    def reset(self):
        """Restart the connection to the Trigno Control Utility server."""
        self._initialize()

    def __del__(self):
        try:
            self._comm_socket.close()
        except:
            pass

    def _send_cmd(self, command):
        self._comm_socket.send(self._cmd(command))
        resp = self._comm_socket.recv(128)
        self._validate(resp)

    @staticmethod
    def _cmd(command):
        return bytes("{}{}".format(command, _BaseTrignoDaq.CMD_TERM),
                     encoding='ascii')

    @staticmethod
    def _validate(response):
        s = str(response)
        if 'OK' not in s:
            print("warning: TrignoDaq command failed: {}".format(s))


class TrignoEMG(_BaseTrignoDaq):
    """
    Delsys Trigno wireless EMG system EMG data.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channel_range : tuple with 2 ints
        Sensor channels to use, e.g. (lowchan, highchan) obtains data from
        channels lowchan through highchan. Each sensor has a single EMG
        channel.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    units : {'V', 'mV', 'normalized'}, optional
        Units in which to return data. If 'V', the data is returned in its
        un-scaled form (volts). If 'mV', the data is scaled to millivolt level.
        If 'normalized', the data is scaled by its maximum level so that its
        range is [-1, 1].
    host : str, optional
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional
        Port of TCU command messages.
    data_port : int, optional
        Port of TCU EMG data access. By default, 50041 is used, but it is
        configurable through the TCU graphical user interface.
    timeout : float, optional
        Number of seconds before socket returns a timeout exception.

    Attributes
    ----------
    rate : int
        Sampling rate in Hz.
    scaler : float
        Multiplicative scaling factor to convert the signals to the desired
        units.
    """

    def __init__(self, channel_range, samples_per_read, units='mV',
                 host='localhost', cmd_port=50040, data_port=50043, timeout=10, freq = 2000):   # host=localhost
        super(TrignoEMG, self).__init__(
            host=host, cmd_port=cmd_port, data_port=data_port,
            total_channels=16, timeout=timeout)

        self.channel_range = channel_range
        self.samples_per_read = samples_per_read

        self.rate = freq

        self.scaler = 1.
        if units == 'mV':
            self.scaler = 1000.
        elif units == 'normalized':
            # max range of EMG data is 11 mV
            self.scaler = 1 / 0.011

    def set_channel_range(self, channel_range):
        """
        Sets the number of channels to read from the device.

        Parameters
        ----------
        channel_range : tuple
            Sensor channels to use (lowchan, highchan).
        """
        self.channel_range = channel_range
        self.num_channels = channel_range[1] - channel_range[0] + 1

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(num_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = super(TrignoEMG, self).read(self.samples_per_read)
        data = data[self.channel_range[0]:self.channel_range[1]+1, :]
        return self.scaler * data


class TrignoAccel(_BaseTrignoDaq):
    """
    Delsys Trigno wireless EMG system accelerometer data.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channel_range : tuple with 2 ints
        Sensor channels to use, e.g. (lowchan, highchan) obtains data from
        channels lowchan through highchan. Each sensor has three accelerometer
        channels.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    host : str, optional
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional
        Port of TCU command messages.
    data_port : int, optional
        Port of TCU accelerometer data access. By default, 50042 is used, but
        it is configurable through the TCU graphical user interface.
    timeout : float, optional
        Number of seconds before socket returns a timeout exception.
    """
    def __init__(self, channel_range, samples_per_read, host='localhost',
                 cmd_port=50040, data_port=50044, timeout=10):
        super(TrignoAccel, self).__init__(
            host=host, cmd_port=cmd_port, data_port=data_port,
            total_channels=48, timeout=timeout)

        self.channel_range = channel_range
        self.samples_per_read = samples_per_read

        self.rate = 148.1

    def set_channel_range(self, channel_range):
        """
        Sets the number of channels to read from the device.

        Parameters
        ----------
        channel_range : tuple
            Sensor channels to use (lowchan, highchan).
        """
        self.channel_range = channel_range
        self.num_channels = channel_range[1] - channel_range[0] + 1

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(num_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = super(TrignoAccel, self).read(self.samples_per_read)
        data = data[self.channel_range[0]:self.channel_range[1]+1, :]
        return data
    
class MVC(TrignoEMG):
    def __init__(self, **kwargs):
        kwargs.setdefault('channel_range', (0, 3))
        kwargs.setdefault('samples_per_read', 200)
        kwargs.setdefault('units', 'mV')

        super().__init__(**kwargs)

        self.num_channels = self.channel_range[1] - self.channel_range[0] + 1
        self.fs = self.rate
    
    def start_mvc_protocol(self, rest_window_sec = 5, contract_window_sec = 3, repetition = 3):
        '''
        Call the function 'perfrom_mvc_protocol' and returns the mean rest periode and peak MVC.
        Write json file with the calibration values for each channel.
        
        args:
            rest_window_sec: Define rest periode as s
                - int
            contract_window_sec: Define contract periode
                - int
            repetition: Number of times to record the sequence of rest and contract movement
                - int
        
        return:
            baseline noise: mean value of rest periode
            MVC: peak value of the MVC periode
        '''
        self.rest_window_sec = int(rest_window_sec)                          
        self.contract_window_sec = int(contract_window_sec)
        self.repetition = int(repetition)
        self.rest_buffer = []
        self.con_buffer = []

        ''' This includes both extensor and flexior movements
        movement = ['flexior', 'extensor']
        for m in movement:
            print(f'Perform {m} movement during calibration')
            self.rest_buffer = numpy.zeros((self.num_channels, rest_window_sec * self.fs * repetition))
            self.con_buffer = numpy.zeros((self.num_channels, contract_window_sec * self.fs * repetition))

            rest_data, contract_data = self.perform_mvc_protocol()

            if m == 'flexior':
                baseline = numpy.mean(rest_data[:, 0:2], axis = 0)            # Calculate mean value for baseline
                peak = numpy.max(contract_data[:, 0:2], axis = 0)                    # Extract highest peak value
            elif m == 'extensor':
                baseline = numpy.mean(rest_data[:, 2:4], axis=0)   # ch2 & ch3
                peak = numpy.max(contract_data[:, 2:4], axis=0)
        
            baseline_noise.append(baseline)
            mvc.append(peak)
        
        # Convert lists to numpy arrays for consistency
        baseline_noise = numpy.array(baseline_noise)
        mvc = numpy.array(mvc)
        '''

        rest_data, contract_data = self.perform_mvc_protocol()

        print(f"rest_data shape: {rest_data.shape}")
        print(f"contract_data shape: {contract_data.shape}")
   
        return rest_data, contract_data
    
    def perform_mvc_protocol(self):
        '''
        args:
            - samples_data:
                TrignoEMG read data
        '''
        self.start()
        print('[Wait] - Flushing buffer until EMG sensor values are non-zero')
        # self.prepare_buffer()

        t0 = time.time()
        print('Protocol starts in 3 sec')
        while (t0 + 3) > time.time():
            self.read()                                         # Remove junk

        for rep in range(self.repetition):
            print(f"Repetition {rep+1}/{self.repetition} | REST")

            start = time.perf_counter()
            while (elapsed := time.perf_counter() - start) < self.rest_window_sec:
                remaining = int(self.rest_window_sec - elapsed)
                print(f"  Rest: {remaining}s remaining", end="\r")

                samples = self.read()       # shape (channels, samples_per_read)
                self.rest_buffer.append(samples)

            
            print("\nCONTRACT!")
            t0 = time.time()                # Flush data
            while (t0 + 2) > time.time():
                self.read()                                         # Remove junk
                
            # ---- CONTRACT PHASE ----
            start = time.perf_counter()
            while (elapsed := time.perf_counter() - start) < self.contract_window_sec:
                remaining = int(self.contract_window_sec - elapsed)
                print(f"  Contract: {remaining}s remaining", end="\r")

                samples = self.read()
                self.con_buffer.append(samples)

            print('Back to rest position')
            t0 = time.time()                # Flush data
            while (t0 + 2) > time.time():
                self.read()                                         # Remove junk
            print("\nCycle complete.\n")

        print("MVC protocol finished.")
        self.con_buffer = numpy.concatenate(self.con_buffer, axis = 1)
        self.rest_buffer = numpy.concatenate(self.rest_buffer, axis = 1)
        return self.rest_buffer.T, self.con_buffer.T
    
    def prepare_buffer(self):
        """
        Wait until the EMG sensor starts streaming non-zero data.
        Useful for discarding initial zero-filled buffers.
        """
        timeout = time.perf_counter() + 5  # seconds
        print("[WAIT] - preparing buffer...")

        while True:
            junk = self.read()

            # Check if data contain any non-zero values
            if numpy.any(junk != 0):
                print("[READY] - data stream active")
                break

            # Timeout check
            if time.perf_counter() > timeout:
                raise TimeoutError("MVC.prepare_buffer: no valid data after 5 seconds")







