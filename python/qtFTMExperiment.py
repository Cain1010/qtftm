# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:23:32 2018
qtFTM Python Module
Author: Zachary S. Buchanan
"""

import struct
import scipy.optimize as spopt
import numpy as np
import scipy.signal as spsig
import shelve
import os
import qtfitting as qtfit
import concurrent.futures
import sys
from scipy.optimize import brentq

class qtFTMExperiment:
    
    _fid_start = 0
    _fid_end = -1
    _zpf = 0
    _ft_min = -1.0
    _ft_max = -1.0
    _ft_winf = 'boxcar'
    
    @classmethod
    def set_ft_defaults(cls,fid_start=None,fid_end=None,zpf=None,ft_min=None,
                        ft_max=None,ft_winf=None):
        """
        Set and store default values for FT
        
        This method stores default arguments for the FT routines in
        persistent, class-scope storage. Class attributes are set for use
        by all instances of BlackChirpExperiment, and the shelve module
        is used to write the values to a file in the user's home directory
        (~/.config/CrabtreeLab/blackchirp-python)
        
        Arguments (all optional; values only set for specified args):
        
            fid_start -- All points before this index (integer) or time
            (float, us) in the FID are set to 0 before FT
            
            fid_end -- All points after this index (integer) or time
            (float, us) in the FID are set to 0 before FT
            
            zpf -- Next power of 2 by which to expand FID length prior to FT
            
            ft_min -- All FT values below this frequency (MHz) are set to 0
            
            ft_max -- All FT values above this frequency (MHz) are set to 0
            
            ft_winf -- Window function applied to FID before FT. This must
            be a value understood by scipy.signal.get_window
        """
        
        #store settings
        with shelve.open(os.path.expanduser('~')
                         +'/.config/CrabtreeLab/qtftm-python') as shf:
                             
            if fid_start is not None:
                cls._fid_start = fid_start
                shf['fid_start'] = fid_start
            if fid_end is not None:
                cls._fid_end = fid_end
                shf['fid_end'] = fid_end
            if zpf is not None:
                cls._zpf = zpf
                shf['zpf'] = zpf
            if ft_min is not None:
                cls._ft_min = ft_min
                shf['ft_min'] = ft_min
            if ft_max is not None:
                cls._ft_max = ft_max
                shf['ft_max'] = ft_max
            if ft_winf is not None:
                cls._ft_winf = ft_winf
                shf['ft_winf'] = ft_winf
        
    
    @classmethod

    def load_settings(cls):
        """
        Read default values from persistent storage.
        
        This method is called when the script is processed. It reads class
        settings from persistent storage, if applicable. This function
        should not need to be called by a script user.
        """
        try:
            with shelve.open(os.path.expanduser('~')
                             +'/.config/CrabtreeLab/qtftm-python') as shf:
                if 'fid_start' in shf:
                    cls._fid_start = shf['fid_start']
                if 'fid_end' in shf:
                    cls._fid_end = shf['fid_end']
                if 'zpf' in shf:
                    cls._zpf = shf['zpf']
                if 'ft_min' in shf:
                    cls._ft_min = shf['ft_min']
                if 'ft_max' in shf:
                    cls._ft_max = shf['ft_max']
                if 'ft_winf' in shf:
                    cls._ft_winf = shf['ft_winf']
        except (OSError, IOError):
            print("No persistent settings found; using defaults")
            pass
        
        cls.print_settings()
            
        
    @classmethod
    def print_settings(cls):
        """
        Print class settings
        """
    
        w = 80
        sb = "".center(w,'*')
        
        print(sb)
        print("qtFTMExperiment Default Settings".center(w))
        print(sb)
        
        labels = []
        values = []
        
        labels.append("FID Start")
        values.append(str(cls._fid_start))
        
        labels.append("FID End")
        values.append(str(cls._fid_end))
        
        labels.append("Zero Pad Factor")
        values.append(str(cls._zpf))
        
        labels.append("FT Min Frequency")
        values.append(str(cls._ft_min))
        
        labels.append("FT Max Frequency")
        values.append(str(cls._ft_max))
        
        labels.append("Window Function")
        values.append(str(cls._ft_winf))
        
        lw = 0
        for s in labels:
            lw = max(len(s),lw)
        vw = 0
        for s in values:
            vw = max(len(s),vw)
        
        strs = []
        for i in range(len(values)):
            strs.append(labels[i].rjust(lw)+" ..."
                        +str(' '+values[i]).rjust(vw,'.'))
        
        for s in strs:
            print(s.center(w))
            
        print("")
        print(str("To modify these values, use qtFTMExperiment." +
              "set_ft_defaults()").center(w))
        print(sb)
        
        
    def __init__(self,number,experimentType='amdor',path=None,quiet=False):
        if number < 1:
            raise ValueError('Experiment number must be greater than 0')
        
        self.d_number = number
        self.quiet = quiet
        
        millions = number // 1000000
        thousands = number // 1000
        
        if path is None:
            path = "/home/data/QtFTM/"+str(experimentType)+'/' + str(millions) \
                   + "/" + str(thousands) 
        
        self.d_header_values = {}
        self.d_header_units = {}
        self.amdoronly0 = [] #rename these to what they actually mean
        self.amdoronly1 = []
        self.amdorfrequencies = []
        self.refNumber = {}
        self.scans = []
        #AMDOR Experiment Parser
        if experimentType == 'amdor':
            with open(path+"/"+str(number)+'.txt') as fid:
                for i in fid:
                    tmp = i.strip().split('\t')
                    if not tmp[0]:
                        continue            
                    if tmp[0][0] == '#':
                        if len(tmp) == 3:
                            self.d_header_values[tmp[0][1:]] = tmp[1]
                            self.d_header_units[tmp[0][1:]] = tmp[2]
                        else:
                            self.d_header_values[tmp[0][1:]] = tmp[1]
                    elif len(tmp) == 2:
                        if tmp[0][:16] != 'amdorfrequencies':
                            if tmp[1] == '0':
                                self.amdoronly0.append(float(tmp[0]))
                            elif tmp[1] == '1':
                                self.amdoronly1.append(float(tmp[0]))
                            self.amdorfrequencies.append(float(tmp[0]))
                    elif tmp[0][:9] != 'amdorscan':
                        scanhdr = {}
                        scanhdr['scan'] = int(tmp[0])
                        scanhdr['iscal'] = bool(int(tmp[1]))
                        scanhdr['isref'] = bool(int(tmp[2]))
                        scanhdr['isval'] = bool(int(tmp[3]))
                        scanhdr['ftid'] = int(tmp[4])
                        scanhdr['drid'] = int(tmp[5])
                        scanhdr['intensity'] = float(tmp[6])
                        scanhdr['elapsedsecs'] = int(tmp[7])
                        tmpscan = scanData(scanhdr)
                        self.scans.append(tmpscan)
                        if scanhdr['isref']:
                            self.refNumber[tmpscan.cavity_freq] = tmpscan.ExpHdr['scan']
            
        self.scanIDoffset = self.scans[0].ExpHdr['scan']
        
    def lookupRefScan(self,refid):
        ref_freq = self.amdorfrequencies[refid]
        ref_scan_number = self.refNumber[ref_freq]
        return(ref_scan_number)
        
    def ft_one(self, index = 0, start = None, end = None, zpf = None, 
               window_f = None, f_min = None, f_max = None):
                   
       f = self.scans[index]
       f_data = np.array(f.fid[:])
       
       s_index = 0
       e_index = f.size
       
       si = self._fid_start
       ei = self._fid_end
       
       if start is not None:
           si = start
       if end is not None:
           ei = end
       
       if type(si) == float:
           s_index = int(si // f.spacing)
       else:
           s_index = si
       
       if type(ei) == float:
           e_index = min((int(ei // f.spacing), f.size))
           if e_index < s_index:
               e_index = f.size
       else:
           if e_index < s_index:
               e_index = f.size
       
       win_size = e_index - s_index
       if window_f is not None:
           f_data[s_index:e_index] *= window_f(win_size)
       else:
           f_data[s_index:e_index] *= spsig.get_window(self._ft_winf, win_size)
       
       if s_index > 0:
           f_data[0:s_index-1] = 0.0
       if e_index < f.size:
           f_data[e_index:] = 0.0
           
       z = self._zpf
       if zpf is not None:
           z = zpf
       if z > 0:
           z = int(z)
           s = 1
           
           while s <= f.size:
               s = s << 1
               
           for i in range(0,z-1):
               s = s << 1
               
           f_data.resize((s),refcheck=False)
           
       ft = np.fft.rfft(f_data)
       nump = len(f_data) // 2 + 1
       
       out_y = np.empty(nump)
       out_x = np.empty(nump)
       df = 1.0 / len(f_data) / f.spacing / 1e6
       
       fn = self._ft_min
       if f_min is not None:
           fn = f_min
       
       fx = self._ft_max
       if f_max is not None:
           fx = f_max
       
       for i in range(0,nump):
           out_x[i] = f.probe_freq+(df*i)
           if ( (fn >= 0.0 and out_x[i] < fn) or (fx >= 0.0 and out_x[i] > fx) ):
               out_y[i] = 0.0
           else:
               out_y[i] = np.absolute(ft[i]) * 1e3
               
       return out_x, out_y       
           
    def analyze_fid(self,xarray,yarray,params):
       
        res = spopt.curve_fit(qtfit.qt_doublegauss,xarray,yarray,
                                  p0=params,full_output=True)
        return res
                        

        
        
        
class scanData:
    def __init__(self,ExpHdr,path=None,start = None, end = None, zpf = None, 
               window_f = None, f_min = None, f_max = None): 
        self.ExpHdr = ExpHdr
        number = self.ExpHdr['scan']
        millions = number // 1000000
        thousands = number // 1000
        d_header_values = {}
        d_header_units = {}
        fid = []
        
        if path is None:
            path = "/home/data/QtFTM/scans/" + str(millions) \
                   + "/" + str(thousands) 
        with open(path+'/'+str(number)+'.txt') as scan:
            for i in scan:
                tmp = i.strip().split('\t')
                if not tmp[0]:
                        continue            
                if tmp[0][0] == '#':
                    if len(tmp) == 3:
                        d_header_values[tmp[0][1:]] = tmp[1]
                        d_header_units[tmp[0][1:]] = tmp[2]
                    elif len(tmp) == 2:
                        d_header_values[tmp[0][1:]] = tmp[1]
                elif tmp[0][:3] == 'fid':
                    continue
                else:
                    fid.append(float(tmp[0]))                                                                                                                                                                                                                                                                                                                                                                                

        
        fid = np.array(fid)
        mean = np.mean(fid)
        self.fid = fid-mean
        self.size = int(d_header_values['FID points'])
        self.DRfreq = float(d_header_values['DR freq'])
        self.spacing = float(d_header_values['FID spacing'])
        self.probe_freq = float(d_header_values['Probe freq'])
        self.cavity_freq = float(d_header_values['Cavity freq'])
            

qtFTMExperiment.load_settings()        
test = qtFTMExperiment(4,'amdor')
x,y = test.ft_one()
#scanTest = scanData(hdr)
        
testfit = spopt.curve_fit(qtfit.qt_doublegauss,x,y,[307,8000,8509.34,0.005,5000,8509.38,0.005])
ymodel = qtfit.qt_doublegauss(xmodel,testfit[0][0],testfit[0][1],testfit[0][2],testfit[0][3],testfit[0][4],testfit[0][5],testfit[0][6])
plt.figure()
plt.plot(x,y)
plt.plot(xmodel,ymodel)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        