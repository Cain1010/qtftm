# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:23:32 2018
qtFTM Python Module
Author: Zachary S. Buchanan
"""

import struct
import numpy as np
import scipy.signal as spsig
import shelve
import os
import bcfitting as bcfit
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
                         +'/.config/CrabtreeLab/blackchirp-python') as shf:
                             
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
                             +'/.config/CrabtreeLab/blackchirp-python') as shf:
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
        print("BlackChirpExperiment Default Settings".center(w))
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
        print(str("To modify these values, use BlackChirpExperiment." +
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
                    elif tmp[0][:9] != 'amdorscan':
                        scanhdr = {}
                        scanhdr['amdorscan'] = int(tmp[0])
                        scanhdr['amdoriscal'] = bool(int(tmp[1]))
                        scanhdr['amdorisref'] = bool(int(tmp[2]))
                        scanhdr['amdorisval'] = bool(int(tmp[3]))
                        scanhdr['amdorftid'] = int(tmp[4])
                        scanhdr['amdordrid'] = int(tmp[5])
                        scanhdr['amdorintensity'] = float(tmp[6])
                        scanhdr['amdorelapsedsecs'] = int(tmp[7])
                        self.scans.append(scanData(scanhdr))
                        

        
        
        
class scanData:
    def __init__(self,ExpHdr,window_f=None,path=None): 
        self.ExpHdr = ExpHdr
        number = self.ExpHdr['amdorscan']
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

        
        raw_data = np.array(fid)
        self.DRfreq = d_header_values['DR freq']
        self.spacing = d_header_values['FID spacing']
        self.Cavityfreq = d_header_values['Cavity freq']
        
        for i in raw_data:
            

        
#test = qtFTMExperiment(3,'amdor')
        
#scanTest = scanData(hdr)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        