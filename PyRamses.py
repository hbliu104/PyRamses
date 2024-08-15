#%%
# Author: Hongbo Liu
# Email: hbliu104@gmail.com

#%%
import serial
import serial.tools.list_ports as serialist

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams['pdf.fonttype'] = 42
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

import time
import glob
import threading
import websockets
import asyncio
import queue
import logging
import csv
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import socket
# ipv6_addr = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6)[0][4][0]

#%%
vals = [2, 4, 8, 9, 10, 12, 16, 20, 24]
types = ['MicroFlu', 'IOM', 'COM', 'IPS', 'SAMIP', 'SCM', 'SAM', 'DFM', 'ADM']

class ws_server:
    def __init__(self) -> None:
        pass

    async def handler(self, ws):
        self.ws = ws
        async for _ in self.ws: pass

    async def send(self, msg):
        if not hasattr(self, 'ws'):
            return
        
        try:
            await self.ws.send(msg)
        except websockets.ConnectionClosedOK:
            print('No active client.')

    async def server(self):
        self.stop = asyncio.get_running_loop().create_future()
        async with websockets.serve(self.handler, host=None, port=5007):
            await self.stop  # run until stop

    def run(self):
        print(f'Thread {threading.get_ident()} start as websocket server.')
        asyncio.run(self.server())
        print(f'Thread {threading.get_ident()} stop as websocket server.')

    def shutdown(self):
        self.stop.set_result(True)
        asyncio.run(self.send(''))


class http_handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        # Cache request
        path = self.path

        # Validate request path, and set type
        if path == "/index.html":
            type = "text/html"
        elif path == "/data.js":
            type = "text/javascript"
        elif path == "/latest.dat":
            type = "text/plain"
        elif path == "/favicon.ico":
            path = "/index.html"
            type = "text/html"
        else:
            # Wild-card/default
            if not path == "/":
                # print("UNRECONGIZED REQUEST: ", path)
                pass

            path = "/index.html"
            type = "text/html"
        
        # Set header with content type
        self.send_response(200)
        self.send_header("Content-type", type)
        self.end_headers()
        
        # Open the file, read bytes, serve
        with open(path[1:], 'rb') as file: 
            self.wfile.write(file.read()) # Send


class http_server():
    def __init__(self) -> None:
        self.port = 3000
        self.ipv4_addr = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)[0][4][0]
        self.server = HTTPServer(('', self.port), http_handler)

    def run(self):
        print(f'Thread {threading.get_ident()} start as http server at {self.ipv4_addr}:{self.port}.')
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            pass
        self.server.server_close()
        print(f'Thread {threading.get_ident()} stop as http server.')


def ramses_parse(filePath, device=None, datatype='CALIBRATED', timestamp=False, ip=False):
    n_test = 0
    flag_ = 0

    dev_list = {}
    if device is None:
        with open(filePath, 'r') as fin:
            for line in fin:
                if line.startswith('IDDevice'):
                    dev_sn = line.split('=')[-1].strip()
                    if dev_sn not in dev_list:
                        dev_list[dev_sn] = {}
                elif line.startswith('IDDataTypeSub1'):
                    data_type = line.split('=')[-1].strip()
                    if data_type in dev_list[dev_sn]:
                        dev_list[dev_sn][data_type] += 1
                    else:
                        dev_list[dev_sn][data_type] = 1
        return dev_list

    with open(filePath, 'r') as fin:
        for line in fin:
            if line.startswith('IDDevice'):
                flag_ = line.split('=')[-1].strip().upper() == device.upper()
            elif line.startswith('IDDataTypeSub1'):
                if line.split('=')[-1].strip() == datatype and flag_:
                    n_test += 1
                flag_ = 0

    if not n_test:
        print('No matched data.')
        return 0

    raw = np.zeros((256, n_test+1))
    time_stamp = []
    inclination = np.zeros(n_test)
    pressure = np.zeros(n_test)

    n = 1
    with open(filePath, 'r') as fin:
        for line in fin:
            if line.startswith('IDDevice'):
                flag_ = line.split('=')[-1].strip().upper() == device.upper()

            elif line.startswith('IDDataTypeSub1'):
                flag_ &= line.split('=')[-1].strip() == datatype

            elif line.startswith('DateTime') and timestamp and flag_:
                time_stamp.append(line.split('=')[-1].strip())

            elif line.startswith('InclV ') and ip and flag_:
                inclination[n-1] = float(line.split('=')[-1].strip())

            elif line.startswith('Pressure') and ip and flag_:
                pressure[n-1] = float(line.split('=')[-1].strip())

            elif line.startswith('[DATA]') and flag_:
                idx = 0
                flag_ <<= 1

            elif line.startswith('[END] of [DATA]'):
                if flag_ >> 1:
                    n += 1
                flag_ = 0

            else:
                if flag_ >> 1:
                    ss = line.lstrip().split(' ')
                    
                    if datatype == 'BACK':
                        raw[idx] = float(ss[1]), float(ss[2])
                    else:
                        if n == 1:
                            raw[idx, 0] = float(ss[0])
                        raw[idx, n] = float(ss[1])
                    idx += 1

    res = (raw[:, 0], raw[:, -1:0:-1])
    
    if timestamp:
        res += (time_stamp[::-1],)

    if ip:
        res += (inclination[::-1], pressure[::-1],)
    
    return res


def bin_replace(bs):
    bs = bs.replace(b'@g', b'\x13') # 0x40 0x67 -> 0x13 Xoff
    bs = bs.replace(b'@f', b'\x11') # 0x40 0x66 -> 0x11 Xon
    bs = bs.replace(b'@e', b'#') # 0x40 0x65 -> 0x23 #
    bs = bs.replace(b'@d', b'@') # 0x40 0x64 -> 0x40 @
    return bs


def serial_rx(ser, q):
    logging.debug(f'Thread {threading.get_ident()} start.')
    while ser.is_open:
        bs = b''
        while ser.in_waiting:
            bs += ser.read(ser.in_waiting)
            time.sleep(0.1)
        if len(bs):
            q.put(bs)
            logging.info('Rx: ' + bs.hex())
        time.sleep(1)
    logging.debug(f'Thread {threading.get_ident()} stop.')


def q_rx(ser, rx_q, devs, plot=False, html=True):
    logging.debug(f'Thread {threading.get_ident()} start.')
    while ser.is_open:
        while rx_q.qsize():
            bs = rx_q.get()
            for dev in devs.values():
                dev.rx_parse(bs, plot=False, html=True)
        time.sleep(1)
    logging.debug(f'Thread {threading.get_ident()} stop.')


def ramses_dev_scan():
    ser_list = []
    ips_chn_list = []
    sn_list = []

    print(time.strftime('%Y%m%d_%H%M%S Serial port scan started.'))
    for dev in serialist.comports():
        print(dev.description)
        with serial.Serial(port=dev.name, baudrate=9600, bytesize=8, parity='N', timeout=1, xonxoff=True) as ser:
            rx_q = queue.Queue()
            sth = threading.Thread(target=serial_rx, args=(ser, rx_q,))
            sth.start()

            ips_chn = '00'
            ss = f'23 {ips_chn} 00 80 B0 00 00 01'
            ser.write(bytes.fromhex(ss))
            logging.debug(f'Tx: ' + ss.replace(' ', ''))
            for _ in range(20):
                if not rx_q.empty():
                    break
                time.sleep(0.1)

            if not rx_q.empty():
                bs = bin_replace(rx_q.get())
                if bs[4] == 255:
                    sn = f'{bs[8]:02X}{bs[7]:02X}'
                    dev_type = types[vals.index(bs[8] >> 3)]
                    print(f'|- {ips_chn} {dev_type} {sn}')
                    if dev_type == 'IPS':
                        for ips_chn in ['02', '04', '06', '08']:
                            ss = f'23 {ips_chn} 00 80 B0 00 00 01'
                            ser.write(bytes.fromhex(ss))
                            logging.debug(f'Tx: ' + ss.replace(' ', ''))
                            for _ in range(20):
                                if not rx_q.empty():
                                    break
                                time.sleep(0.1)
                            if not rx_q.empty():
                                bs = bin_replace(rx_q.get())
                                if bs[4] == 255:
                                    sn = f'{bs[8]:02X}{bs[7]:02X}'
                                    dev_type = types[vals.index(bs[8] >> 3)]
                                    print(f'   |- {ips_chn} {dev_type} {sn}')
                                    ser_list.append(dev.name)
                                    ips_chn_list.append(ips_chn)
                                    sn_list.append(sn)
                    else:
                        ser_list.append(dev.name)
                        ips_chn_list.append(ips_chn)
                        sn_list.append(sn)

    print(time.strftime('%Y%m%d_%H%M%S Serial port scan finished.'))
    return list(zip(ser_list, ips_chn_list, sn_list))


class RAMSES:
    def __init__(self, config_path='./Ramses Calibration_2017/', sn=None) -> None:
        if not hasattr(RAMSES, 'config_path'):
            RAMSES.config_path = config_path
        
        if sn is not None:
            self.sn = sn
            self.config_parse()

        self.measuring = threading.Event()

    def __repr__(self) -> str:
        return self.sn

    def config_parse(self):
        config_ini = glob.glob(RAMSES.config_path + self.sn + '/*.ini')[0]

        param = {}
        with open(config_ini) as fin:
            dev_id = -1
            for line in fin:
                if line.upper().startswith('[DEVICE]'):
                    dev_id += 1
                    param[f'dev_{dev_id}'] = {}
                if line.find('=') > 0:
                    a, b = line.split('= ')
                    if b != '\n':
                        param[f'dev_{dev_id}'][a.rstrip()] = b.rstrip()
            param['n_dev'] = dev_id + 1

        for dev in [param[f'dev_{x}'] for x in range(param['n_dev'])]:
            if dev['IDDeviceType'] == 'SAM':
                self.id_dev = dev['IDDevice']
                c0s = float(dev['c0s'])
                c1s = float(dev['c1s'])
                c2s = float(dev['c2s'])
                c3s = float(dev['c3s'])
                self.sam_type = dev['IDDeviceTypeSub1']
                self.wl = c0s + np.arange(2, 257) * c1s + np.power(np.arange(2, 257), 2) * c2s + np.power(np.arange(2, 257), 3) * c3s
                self.t0 = 8192
                self.DarkPixelStart = int(dev['DarkPixelStart'])
                self.DarkPixelStop = int(dev['DarkPixelStop'])
                self.bkg1, self.bkg2 = ramses_parse(f'./Ramses Calibration_2017/{self.sn}/Back_{self.id_dev}.dat', self.id_dev, datatype='BACK')
                self.air = ramses_parse(f'./Ramses Calibration_2017/{self.sn}/Cal_{self.id_dev}.dat', self.id_dev, datatype='CAL')[1]
                self.aqua = ramses_parse(f'./Ramses Calibration_2017/{self.sn}/CalAQ_{self.id_dev}.dat', self.id_dev, datatype='CAL')[1]

                if not hasattr(self, 'dev_type'):
                    self.dev_type = 'SAM'

            elif dev['IDDeviceType'] == 'IP':
                self.Incl_Xoffset = int(dev['Incl_Xoffset'])
                self.Incl_Yoffset = int(dev['Incl_Yoffset'])
                self.Incl_Xgain = float(dev['Incl_Xgain'])
                self.Incl_Ygain = float(dev['Incl_Ygain'])
                self.Incl_Kref = float(dev['Incl_Kref'])
                self.press_sens = float(dev['Press_Sens_mV_bar_4mA'])
                if self.press_sens <= 0:
                    self.press_sens = 4 * float(dev['Press_Sens_mV_bar_1mA'])
                self.Press_Gain = float(dev['Press_Gain'])
                
                self.dev_type = 'SAMIP'

    def attach(self, ser, ips_chn, sn, interval_s=0, n_total=1):
        self.ser = ser
        self.ips_chn = ips_chn
        self.interval_s = interval_s
        self.n_total = n_total
        if hasattr(self, 'sn'):
            if self.sn != sn:
                print('Device does not match.')
        else:
            self.sn = sn
            self.config_parse()

    def tx_query(self):
        ss = f'23 {self.ips_chn} 00 80 B0 00 00 01'
        self.ser.write(bytes.fromhex(ss))
        logging.debug(f'[{self.sn}] Tx: ' + ss.replace(' ', ''))

    def tx_set_t(self, t_hex='00'):
        if self.dev_type == 'SAMIP':
            ss = f'23 {self.ips_chn} 00 30 78 05 {t_hex} 01'
        else:
            ss = f'23 {self.ips_chn} 00 80 78 05 {t_hex} 01'
        self.ser.write(bytes.fromhex(ss))
        logging.debug(f'[{self.sn}] Tx: ' + ss.replace(' ', ''))

    def tx_measure(self):
        ss = f'23 {self.ips_chn} 00 80 A8 00 81 01'
        self.ser.write(bytes.fromhex(ss))
        logging.info(f'[{self.sn}] Tx: ' + ss.replace(' ', ''))
        self.t_trigger = time.strftime('%Y%m%d_%H%M%S')
        self.measuring.set()

    def rx_parse(self, bs, corr=False, plot=False, html=False):
        bs_seg = [b'#' + x for x in bs.split(b'#')][1:]
        raw = []
        for bs in bs_seg:
            bs = bin_replace(bs)
            # n_byte = 2 ** ((bs[1] >> 5) + 1)
            # print(f'n_byte: {n_byte}')            
            if bs[1] & 0b1111 != int(self.ips_chn): # dev_id
                continue

            if bs[4] == 255: # info
                # print(f'dev_id2: {bs[2]:02x}')
                # print(f'module_id: {bs[3]:02x}')
                # print(f'time1: {bs[5]:x}')
                # print(f'time2: {bs[6]:x}')
                print(f'\npkg_type: info')
                print(f'sn: {bs[8]:02X}{bs[7]:02X}')
                print(f'dev_type: {types[vals.index(bs[8] >> 3)]}')
                print(f'firmware: {bs[10]:x}.{bs[9]:02x}')
                # print(f'rest: {bs[7+4:7+n_byte].hex()}')
                continue
            elif bs[4] == 254: # error
                print(f'\npkg_type: error')
                return
            else: # data
                pass

            if bs[3] == 0x30 or bs[3] == 0x00: # SAM
                if bs[4] == 0x07:
                    t_hex = bs[7]
                    raw = []
                data_frame = bs[7:]
                raw = raw + [int.from_bytes(data_frame[x:x+2], 'little') for x in range(0, len(data_frame)-2, 2)]
            elif bs[3] == 0x20: # IP
                x = (bs[11] - self.Incl_Xoffset) * self.Incl_Xgain
                y = (bs[12] - self.Incl_Yoffset) * self.Incl_Ygain
                self.inclination = 180/np.pi * np.arctan(np.sqrt(np.tan(x * np.pi / 180)**2 + np.tan(y * np.pi / 180)**2)) # [degree]
                
                npress = int.from_bytes(bs[13:15], 'little')
                nbg = int.from_bytes(bs[17:19], 'little')
                nrefh = int.from_bytes(bs[19:21], 'little')
                nrefl = int.from_bytes(bs[21:23], 'little')
                noffset = nrefl - self.Incl_Kref * (nrefh - nrefl)
                vpress = self.Incl_Kref * (npress - noffset) / (nbg - noffset)
                self.p_bar = 1000 * vpress / (self.press_sens * self.Press_Gain)
                self.d_water = (self.p_bar - 1.021) * 10 # [m]

            else:
                print(f'Device type {bs[3]} is not SAM or IP.')

        if len(raw) == 256:
            self.measuring.clear()

            raw[0] = t_hex
            self.t = 2 * 2 ** t_hex
            self.raw = raw

            dst_ = np.array(raw, dtype=float).reshape((-1, 1)) / 65535 - (self.bkg1.reshape((-1, 1)) + self.t/self.t0 * self.bkg2)
            if corr:
                noise = np.maximum(0, dst_[self.DarkPixelStart:self.DarkPixelStop+1].mean())
            else:
                noise = dst_[self.DarkPixelStart:self.DarkPixelStop+1].mean()
            self.cali = (dst_ - noise) * self.t0 / self.t / self.air
            self.cali[0] = t_hex

            if not self.ser is None:
                logging.info(f'[{self.sn}] READY')

            if plot:
                self.fig_plot()

            if html:
                self.html_preview()
                self.dat_formatter()

        else:
            if len(raw):
                print('Incomplete data.')

    def html_preview(self):
        self.msg = json.dumps({"dev":self.sn,
                               "sam_type":self.sam_type,
                               "t_trigger":self.t_trigger,
                               "t_exp":self.t,
                               "data":{'x': self.wl.tolist(), 'y': self.cali[1:].flatten().tolist(), 'name': self.sn, 'type': 'scatter'}}).replace('NaN', 'null')

    def fig_plot(self):
        plt.figure()
        plt.plot(self.wl, self.cali[1:], label=self.sn)
        plt.grid()
        plt.ylim(bottom=0)
        plt.xlabel('Wavelength (nm)')
        if self.sam_type.startswith('ARC'):
            plt.ylabel(r'Radiance (mW$\cdot$m$^{-2}\cdot$sr$^{-1}\cdot$nm$^{-1}$)')
        else:
            plt.ylabel(r'Irradiance (mW$\cdot$m$^{-2}\cdot$nm$^{-1}$)')
        
        if hasattr(self, 'inclination'):
            plt.text(0.6, 0.6, f'Incl. = {self.inclination:.2f}' + r'$^{\circ}$' + f'\nPressure: {self.p_bar:.2f} bar\nWater depth: {self.d_water:.2f} m', transform=plt.gca().transAxes)
        
        if hasattr(self, 't_trigger'):
            plt.title(f'{self.t_trigger}, t = {self.t} ms')
        else:
            plt.title(f't = {self.t} ms')

        plt.legend()
        plt.tight_layout()

        if not self.ser is None:
            plt.savefig(f'./img_{self.sn}.png')
            plt.close()
        else:
            plt.show()

        print(f'./img_{self.sn}.png saved.')


    def dat_formatter(self, dat_path='./latest.dat'):
        dev_info = {'Version':'1', 'IDData':'', 'IDDevice':self.id_dev, 'IDDataType':'SPECTRUM', 'IDDataTypeSub1':'', 'IDDataTypeSub2':'', 'IDDataTypeSub3':'', 'DateTime':time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(self.t_trigger, '%Y%m%d_%H%M%S')), 'PositionLatitude':'0', 'PositionLongitude':'0', 'Comment':'', 'CommentSub1':'', 'CommentSub2':'', 'CommentSub3':'', 'IDMethodType':f'{self.dev_type} Control', 'MethodName':f'{self.dev_type}_{self.sn}', 'Mission':'No Mission', 'MissionSub':'1', 'RecordType':'0'}

        dev_attr = {'CalFactor':'1', 'IDBasisSpec':'', 'IDDataBack':'', 'IDDataCal':'', 'IntegrationTime':self.t, 'P31':'-1', 'P31e':'0', 'PathLength':'+INF', 'PathLengthCustomOn':'0', 'RAWDynamic':'65535', 'Salinity':'0', 'Temperature':'-NAN', 'Unit1':'', 'Unit2':'', 'Unit3':'', 'Unit4':'$f1 $00 Status', 'p999':'1'}

        if self.dev_type == 'SAMIP':
            try:
                dev_attr['PressValid'] = 1
                dev_attr['Pressure'] = self.p_bar
                dev_attr['InclValid'] = 1
                dev_attr['InclV'] = self.inclination
            except AttributeError:
                pass

        # raw
        dev_info['IDDataTypeSub1'] = 'RAW'
        dev_attr['Unit1'] = '$05 $00 Pixel'
        dev_attr['Unit2'] = '$03 $05 Intensity counts'
        dev_attr['Unit3'] = '$f0 $05 Error counts'
        
        with open(dat_path, 'a') as fout:
            _ = fout.write('[Spectrum]\n')
            for key, val in dev_info.items():
                _ = fout.write('{:<19s}= {}\n'.format(key, val))

            _ = fout.write('[Attributes]\n')
            for key, val in dev_attr.items():
                _ = fout.write('{} = {}\n'.format(key, val))
            _ = fout.write('[END] of [Attributes]\n')
            
            _ = fout.write('[DATA]\n')
            for idx, val in enumerate(self.raw):
                _ = fout.write(f' {idx} {val} 0 0\n')
            _ = fout.write('[END] of [DATA]\n')
            _ = fout.write('[END] of Spectrum\n\n')
        
        # calibrated
        dev_info['IDDataTypeSub1'] = 'CALIBRATED'
        dev_attr['Unit1'] = '$01 $01 Wavelength nm'
        if self.sam_type.startswith('ARC'):
            dev_attr['Unit2'] = '$03 $03 Intensity mW/(m^2 nm Sr)'
            dev_attr['Unit3'] = '$f0 $03 Error mW/(m^2 nm Sr)'
        else:
            dev_attr['Unit2'] = '$03 $06 Intensity mW/(m^2 nm)'
            dev_attr['Unit3'] = '$f0 $06 Error mW/(m^2 nm)'
        
        with open(dat_path, 'a') as fout:
            _ = fout.write('[Spectrum]\n')
            for key, val in dev_info.items():
                _ = fout.write('{:<19s}= {}\n'.format(key, val))

            _ = fout.write('[Attributes]\n')
            for key, val in dev_attr.items():
                _ = fout.write('{} = {}\n'.format(key, val))
            _ = fout.write('[END] of [Attributes]\n')
            
            _ = fout.write('[DATA]\n')
            _ = fout.write(f' 0 {self.raw[0]} 0 0\n')
            for wl, val in zip(self.wl, self.cali[1:, 0]):
                _ = fout.write(f' {wl} {val} 0 0\n')
            _ = fout.write('[END] of [DATA]\n')
            _ = fout.write('[END] of Spectrum\n\n')


def log2dat(log_path, dat_path=None, corr=True):
    if dat_path is None:
        dat_path = log_path.split('.log')[0] + '.dat'

    with open(dat_path, 'w') as _:
        pass
    
    ips_chn_list = []
    sn_list = []
    with open(log_path, 'r') as fin:
        for line in fin:
            if line.find('Tx') > 0:
                sn_list.append(line.split('[')[-1][:4])
                ips_chn_list.append(line.split('Tx: ')[-1][2:4])
    ramses_dev_list = list(set(zip(ips_chn_list, sn_list)))

    ramses_dev = {}
    dev_flag = {}
    for idx in range(len(ramses_dev_list)):
        ips_chn, sn = ramses_dev_list[idx]
        ramses_dev[sn] = RAMSES(sn=sn)
        ramses_dev[sn].attach(ser=None, ips_chn=ips_chn, sn=sn)
        dev_flag[sn] = False
    

    with open(log_path, 'r') as fin:
        for line in fin:
            if line.find('READY') > 0:
                for sn, dev in ramses_dev.items():
                    if sn == line.split('[')[-1].split(']')[0] and dev_flag[sn]:
                        dev.dat_formatter(dat_path=dat_path)
                        dev_flag[sn] = False
            elif line.find('Tx') > 0:
                for sn, dev in ramses_dev.items():
                    if sn == line.split('[')[-1].split(']')[0]:
                        dev_flag[sn] = True
                        dev.t_trigger = line.split(' ')[0]
            else:
                bs = bytes.fromhex(line.split('Rx: ')[-1].strip())
                for sn, dev in ramses_dev.items():
                    if dev_flag[sn]:
                        dev.rx_parse(bs, corr=corr)
    
    return [x.id_dev for x in ramses_dev.values()]


def dat2csv(dat_path, datatype='CALIBRATED'):
    dev_list = ramses_parse(dat_path)

    for dev_id in dev_list.keys():
        wl, val, ts = ramses_parse(dat_path, dev_id, datatype=datatype, timestamp=True)

        csv_path = dat_path.split('.dat')[0] + f'_{dev_id}_.csv'
        with open(csv_path, 'w', newline='') as fout:
            csv_writer = csv.writer(fout)

            csv_writer.writerow(wl)
            csv_writer.writerows(np.hstack((np.array(ts).reshape((-1, 1)), val[1:].T)))

def raw2cal(dat_path, cal_coeff, corr=False, plot=False):
    for dev_id in ramses_parse(dat_path).keys():
        sn = dev_id.split('_')[-1]
        dev = RAMSES(sn=sn)
        wl_idx, raw, ts = ramses_parse(dat_path, dev_id, datatype='RAW', timestamp=True)

        dev.t = 2 * 2**raw[0]

        dst_ = raw / 65535 - (dev.bkg1.reshape((-1, 1)) + dev.t/dev.t0 * dev.bkg2)
        if corr:
            noise = np.maximum(0, dst_[dev.DarkPixelStart:dev.DarkPixelStop+1].mean())
        else:
            noise = dst_[dev.DarkPixelStart:dev.DarkPixelStop+1].mean()

        if cal_coeff == 'air':
            dev.cali = (dst_ - noise) * dev.t0 / dev.t / dev.air
        elif cal_coeff == 'water':
            dev.cali = (dst_ - noise) * dev.t0 / dev.t / dev.aqra
        dev.cali[0] = raw[0]

        csv_path = dat_path.split('.dat')[0] + f'_{dev_id}_calibrated.csv'
        with open(csv_path, 'w', newline='') as fout:
            csv_writer = csv.writer(fout)

            csv_writer.writerow(np.hstack((0, dev.wl)))
            csv_writer.writerows(np.hstack((np.array(ts).reshape((-1, 1)), dev.cali[1:].T)))

        if plot:
            # RAW
            plt.figure()
            _ = plt.semilogy(wl_idx[1:], raw[1:, 0])
            plt.title('RAW')
            plt.grid(True)
            plt.xlim(1, 256)

            # CAL
            plt.figure()
            wl, val = ramses_parse(dat_path, dev_id, datatype='CALIBRATED', timestamp=False)
            plt.semilogy(dev.wl, dev.cali[1:, 0], label='manual')
            plt.semilogy(wl[1:], val[1:, 0], label='auto')
            plt.title('CALIBRATED')
            plt.legend()
            plt.grid(True)
            plt.show()


def json_parse(json_path='./schedule.json'):
    with open(json_path, 'r') as fin:
        schedule = json.load(fin)
    
    return [list(x.values()) for x in schedule['Devices']]


#%%
if __name__ == "__main__":
    matplotlib.use('agg')
    log_path = time.strftime('./ramses_%Y%m%d_%H%M%S.log')
    logging.basicConfig(filename=log_path, filemode='w', format='%(asctime)s %(message)s', datefmt='%Y%m%d_%H%M%S', level=logging.INFO)

    ramses_dev_list = ramses_dev_scan()

    input('')
    if len(ramses_dev_list):
        with serial.Serial(port=ramses_dev_list[0][0], baudrate=9600, bytesize=8, parity='N', timeout=1, xonxoff=True) as ser:

            rx_q = queue.Queue()
            rx_listen = threading.Thread(target=serial_rx, args=(ser, rx_q,))
            rx_listen.start()

            ramses_dev = {}
            for idx in range(len(ramses_dev_list)):
                _, ips_chn, sn = ramses_dev_list[idx]
                ramses_dev[f'dev_{idx}'] = RAMSES(config_path='./Ramses Calibration_2017/', sn=sn)
                ramses_dev[f'dev_{idx}'].attach(ser=ser, ips_chn=ips_chn, sn=sn)

            q_listen = threading.Thread(target=q_rx, args=(ser, rx_q, ramses_dev))
            q_listen.start()
            
            while True:
                if input('Start measurement? ') != 'n':
                    logging.info('NEW')
                    for dev in ramses_dev.values():
                        dev.tx_set_t(t_hex='00')
                        dev.tx_measure()
                    input('')
                else:
                    logging.info('END')
                    break
                    
            print(time.strftime('%Y%m%d_%H%M%S Finished.'))

    ramses_id_dev_list = log2dat(log_path)

    if len(ramses_id_dev_list):
        dat_path = log_path.split('.log')[0] + '.dat'

        dat2csv(dat_path)
        
        matplotlib.use('TkAgg')
        for id_dev in ramses_id_dev_list:
            wl, res = ramses_parse(dat_path, id_dev)
            
            plt.figure()
            plt.plot(wl[1:], res[1:, :])
            plt.grid()
            plt.ylim(bottom=0)
            plt.xlabel('Wavelength (nm)')
            plt.title(id_dev)
        plt.show()
