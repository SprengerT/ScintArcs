import numpy as np
from numpy import newaxis as na
import os
from numpy.ctypeslib import ndpointer
import ctypes
import scipy
from scipy import odr
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

dir_path = os.path.dirname(os.path.realpath(__file__))
file_c = os.path.join(os.path.join(dir_path,"libcpp"),"lib_NuT.so")
lib = ctypes.CDLL(file_c)

#load C++ library for fast NuT transform
lib.NuT.argtypes = [
    ctypes.c_int,   # N_t
    ctypes.c_int,   # N_nu
    ctypes.c_int,   # N_fD
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # tt [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # fD [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # DS [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_real [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_im [N_t*N_nu]
] 

class SecSpec():
    
    def __init__(self,data_fD,data_tau,data_SS,kwargs):
        #specifications
        self.doppler_scale = kwargs.get("doppler_scale",0.001)
        self.delay_scale = kwargs.get("delay_scale",1.0e-06)
        self.xmin = kwargs.get("xmin",-10.0)
        self.xmax = kwargs.get("xmax",10.0)
        self.ymin = kwargs.get("ymin",0.0)
        self.ymax = kwargs.get("ymax",6.0)
        self.title = kwargs.get("title",'Secondary Spectrum')
        self.vmin = kwargs.get("vmin",5.2)
        self.vmax = kwargs.get("vmax",7.7)
        self.sigma_tau_max = (self.ymax-self.ymin)/8.
        self.sigma_fD_max = (self.xmax-self.xmin)/8.
        
        #reduce data
        self.data_fD,self.data_tau,self.data_SS = self.reduce_data(data_fD,data_tau,data_SS)
        
    def draw_SS(self,figure,ax):
        #specifications
        cmap = 'Greys'
        vmin = self.vmin
        vmax = self.vmax
        xlabel = '$f_D$ [mHz]'
        ylabel = r'$\tau$ [$\mu$s]'
        
        # #clear the axis
        # ax.clear()
        
        #draw the plot
        ax.set_title(self.title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim([self.xmin,self.xmax])
        ax.set_ylim([self.ymin,self.ymax])
        ax.pcolormesh(self.data_fD,self.data_tau,np.swapaxes(self.data_SS,0,1),cmap=cmap,vmin=vmin,vmax=vmax,shading='nearest')
        figure.canvas.draw_idle()
        
    def reduce_data(self,data_fD,data_tau,data_SS):
        #prepare data
        # - apply scale
        data_fD = data_fD/self.doppler_scale
        data_tau = data_tau/self.delay_scale
        # - cut off out of range data
        min_index_t = 0
        max_index_t = len(data_fD)-1
        for index_t in range(len(data_fD)):
            if data_fD[index_t]<self.xmin:
                min_index_t = index_t
            elif data_fD[index_t]>self.xmax:
                max_index_t = index_t
                break
        data_fD = data_fD[min_index_t:max_index_t+1]
        min_index_nu = 0
        max_index_nu = len(data_tau)-1
        for index_nu in range(len(data_tau)):
            if data_tau[index_nu]<self.ymin:
                min_index_nu = index_nu
            elif data_tau[index_nu]>self.ymax:
                max_index_nu = index_nu
                break
        data_tau = data_tau[min_index_nu:max_index_nu+1]
        # - load spectrum
        data_SS = np.abs(data_SS[min_index_t:max_index_t+1,min_index_nu:max_index_nu+1])
        # - downsample
        N_fD = len(data_fD)
        N_tau = len(data_tau)
        sampling_fD = np.max([int(N_fD/400),1])
        sampling_tau = np.max([int(N_tau/400),1])
        data_SS = block_reduce(data_SS, block_size=(sampling_fD,sampling_tau), func=np.mean)
        coordinates = np.array([data_fD,data_fD])
        coordinates = block_reduce(coordinates, block_size=(1,sampling_fD), func=np.mean, cval=data_fD[-1])
        data_fD = coordinates[0,:]
        coordinates = np.array([data_tau,data_tau])
        coordinates = block_reduce(coordinates, block_size=(1,sampling_tau), func=np.mean, cval=data_tau[-1])
        data_tau = coordinates[0,:]
        # - apply logarithmic scale
        min_nonzero = np.min(data_SS[np.nonzero(data_SS)])
        data_SS[data_SS == 0] = min_nonzero
        data_SS = np.log10(data_SS)
        #print(data_SS.max(),data_SS.min())
        return data_fD,data_tau,data_SS
        
class manual_data():
    def __init__(self,figure,ax,xmin,xmax):
        self.points_x = []
        self.points_y = []
        self.xerr = 1.
        self.yerr = 1.
        self.ax = ax
        self.figure = figure
        
        self.plot_points = ax.errorbar(self.points_x,self.points_y,xerr=self.xerr,yerr=self.yerr,color='red',linestyle='',marker='o',markersize=0, fillstyle='none',alpha=0.5)
        
        #initialize parabolas
        #xmin = -11.
        #xmax = 11.
        N_x = 100
        self.x_par = np.linspace(xmin,xmax,num=N_x,dtype=float)
        self.y_par = np.zeros(N_x,dtype=float)
        self.y_par_l = np.zeros(N_x,dtype=float)
        self.y_par_r = np.zeros(N_x,dtype=float)
        self.parabola, = ax.plot(self.x_par,self.y_par,color='red',linestyle='-',markersize=0)
        self.parabola_l, = ax.plot(self.x_par,self.y_par_l,color='red',linestyle='--',markersize=0)
        self.parabola_r, = ax.plot(self.x_par,self.y_par_r,color='red',linestyle='--',markersize=0)
        
        self.figure.canvas.draw_idle()
        
    def reset(self,figure,ax,xmin,xmax):
        self.__init__(figure,ax,xmin,xmax)
        
    def onclick(self,event):
        if str(event.button)=='MouseButton.RIGHT':
            fD = event.xdata
            tau = event.ydata
            self.points_x.append(fD)
            self.points_y.append(tau)
            self.update_error(self.xerr,self.yerr)
            self.figure.canvas.draw_idle()
            print("Added point at ({0},{1}).".format(fD,tau))
         
    def update_error(self,fD_err,tau_err):
        self.xerr = fD_err
        self.yerr = tau_err
        x = np.array(self.points_x)
        y = np.array(self.points_y)
        xerr = self.xerr
        yerr = self.yerr
        plotline, caplines, barlinecols = self.plot_points
        # Replot the data first
        plotline.set_data(x,y)

        # Update the error bars
        barlinecols[0].set_segments(zip(zip(x-xerr,y), zip(x+xerr,y)))
        barlinecols[1].set_segments(zip(zip(x,y-yerr), zip(x,y+yerr)))
        
    def undo(self,event):
        del self.points_x[-1]
        del self.points_y[-1]
        self.update_error(self.xerr,self.yerr)
        self.figure.canvas.draw_idle()
        
    def fit(self,event):
        print("Fitting...")
        def fitfunc(beta,data):
            return beta[0]*data**2
            
        data = odr.RealData(self.points_x, self.points_y, self.xerr, self.yerr)
        model = odr.Model(fitfunc)
        
        fitter = odr.ODR(data, model, [1.])
        fitter.set_job(fit_type=0)
        output = fitter.run()
        self.eta = output.beta[0]
        self.eta_err = output.sd_beta[0]
        
        self.y_par = fitfunc([self.eta],self.x_par)
        self.y_par_l = fitfunc([self.eta-self.eta_err],self.x_par)
        self.y_par_r = fitfunc([self.eta+self.eta_err],self.x_par)
        self.parabola.set_ydata(self.y_par)
        self.parabola_l.set_ydata(self.y_par_l)
        self.parabola_r.set_ydata(self.y_par_r)
        self.figure.canvas.draw_idle()
        print("Fit result: eta={0} +- {1}.".format(self.eta,self.eta_err))

def compute_nutSS(t,nu,DS,nu0=1.4e+9):
    N_t = len(t)
    N_nu = len(nu)
    dt = np.diff(t).mean()
    dnu = np.diff(nu).mean()
    t0 = np.mean(t)
    #compute axes
    fD = np.fft.fftshift(np.fft.fftfreq(N_t,dt))
    tau = np.fft.fftshift(np.fft.fftfreq(N_nu,dnu))
    #- prepare data
    data = DS - np.mean(DS)
    tt = (t-t0)/nu0
    hss_real = np.zeros((N_t*N_nu),dtype='float64')
    hss_im = np.zeros((N_t*N_nu),dtype='float64')
    lib.NuT(N_t,N_nu,N_t,tt.astype('float64'),nu.astype('float64'),fD.astype('float64'),data.astype('float64').flatten(),hss_real,hss_im)
    hss = hss_real.reshape((N_t,N_nu))+1.j*hss_im.reshape((N_t,N_nu))
    SS = np.abs(np.fft.fftshift(np.fft.fft(hss,axis=1),axes=1))**2
    
    return fD,tau,SS

def compute_staufD(fD,tau,SS,**kwargs):
    N_stau = kwargs.get("N_stau",100)
    N_fD = len(fD)
    tau_max = kwargs.get("tau_max",np.max(tau))
    rnoise = kwargs.get("remove_noise",True)
    data = np.sqrt(SS)
    dtau = np.diff(tau).mean()
    
    #preparations
    noise = np.median(data)
    stau_max = np.sqrt(tau_max)
    stau = np.linspace(-stau_max,stau_max,num=N_stau,endpoint=True)
    dstau = stau[1]-stau[0]
    stau_u = stau+dstau/2.
    stau_l = stau-dstau/2.
    taus = stau**2*np.sign(stau)
    taus_1 = stau_u**2*np.sign(stau_u)
    taus_2 = stau_l**2*np.sign(stau_l)
    taus_u = np.maximum(taus_1,taus_2)
    taus_l = np.minimum(taus_1,taus_2)
    # - get pixel boundaries of SS
    ltau = tau - dtau/2.
    rtau = tau + dtau/2.
    
    #create containers
    staufD = np.zeros((N_fD,N_stau),dtype=float)
    
    #perform the computation
    # - main computation
    for i_stau in range(N_stau):
        # - determine indices of boundary pixels
        i_tau_l = np.argmax(rtau>taus_l[i_stau])
        i_tau_u = np.argmax(rtau>taus_u[i_stau])
        # - sum pixels
        for i_tau in range(i_tau_l,i_tau_u+1):
            length = (np.min([taus_u[i_stau],rtau[i_tau]])-np.max([taus_l[i_stau],ltau[i_tau]]))/dtau
            staufD[:,i_stau] += data[:,i_tau]*length
            if rnoise:
                staufD[:,i_stau] -= noise*length
                
    return stau, fD, staufD

def ParabolicFitter(fD,tau,SS,**kwargs):
    #set up matplotlib
    labelsize = 12
    textsize = 10
    pgf_with_pdflatex = {
    "font.size": labelsize,          
    "axes.labelsize": labelsize,               # LaTeX default is 10pt font. 
    "axes.titlesize": labelsize,
    "legend.fontsize": textsize,
    "xtick.labelsize": textsize,
    "ytick.labelsize": textsize,
    }
    mpl.rcParams.update(pgf_with_pdflatex) 
    
    #set up the canvas
    plot_width = 1000
    plot_height = 700
    plot_dpi = 100
    plot_bottom = 0.10
    plot_top = 0.95
    plot_left = 0.06
    plot_right = 0.65
    plot_wspace = 0.2
    plot_hspace = 0.2
    
    #create figure and axes
    figure = plt.figure(figsize=(plot_width/plot_dpi,plot_height/plot_dpi),dpi=plot_dpi)
    plt.subplots_adjust(bottom=plot_bottom,top=plot_top,left=plot_left,right=plot_right,wspace=plot_wspace,hspace=plot_hspace)
    ax_SS = figure.add_subplot(1,1,1)
    SS = SecSpec(fD,tau,SS,kwargs)
    SS.draw_SS(figure,ax_SS)
    global errdots
    errdots = manual_data(figure,ax_SS,SS.xmin,SS.xmax)
    
    #create widgets
    # - compute locations
    xmin = 0.7
    ymin = 0.6
    xmax = 0.95
    ymax = 0.95
    nrows = 5
    ncols = 1
    yspace = 0.02
    xspace = 0.02
    xfullwidth = xmax-xmin
    yfullwidth = ymax-ymin
    xwidth = (xfullwidth-(ncols-1)*yspace)/ncols
    ywidth = (yfullwidth-(nrows-1)*yspace)/nrows
    xpos = [xmin+i*(xwidth+xspace) for i in range(ncols)]
    ypos = [ymin+i*(ywidth+yspace) for i in range(nrows)]
    # - place widgets
    button_fit = mpl.widgets.Button(plt.axes([xpos[0],ypos[4],xwidth,ywidth]), "fit")
    button_save = mpl.widgets.Button(plt.axes([xpos[0],ypos[3],xwidth,ywidth]), "save")
    button_undo = mpl.widgets.Button(plt.axes([xpos[0],ypos[2],xwidth,ywidth]), "undo")
    slider_fD_err = mpl.widgets.Slider(plt.axes([xpos[0],ypos[1],xfullwidth,ywidth]),r'$\sigma(f_\mathrm{D})$',0.,SS.sigma_fD_max,valinit=0.)
    slider_tau_err = mpl.widgets.Slider(plt.axes([xpos[0],ypos[0],xfullwidth,ywidth]),r'$\sigma(\tau)$',0.,SS.sigma_tau_max,valinit=0.)
    # - define functions of widgets
    def fct_button_save(event):
        plt.close()
    def update_error(event):
        errdots.update_error(slider_fD_err.val,slider_tau_err.val)
    # - connect widgets to methods/functions
    figure.canvas.mpl_connect('button_press_event', errdots.onclick)
    button_fit.on_clicked(errdots.fit)
    button_save.on_clicked(fct_button_save)
    button_undo.on_clicked(errdots.undo)
    slider_fD_err.on_changed(update_error)
    slider_tau_err.on_changed(update_error)
    
    plt.show()
    eta = errdots.eta
    eta_err = errdots.eta_err
    return eta,eta_err

def LineFinder(stau, fD, staufD,**kwargs):
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    xmin = kwargs.get("xmin",np.min(fD)/1.0e-3)
    xmax = kwargs.get("xmax",np.max(fD)/1.0e-3)
    ymin = kwargs.get("ymin",np.min(stau)/1.0e-3)
    ymax = kwargs.get("ymax",np.max(stau)/1.0e-3)
    nu0 = kwargs.get("nu0",1.4e+9)
    zeta_max = kwargs.get("zeta_max",1.0e-9)
    zeta_init= kwargs.get("zeta_init",0.0)

    def plot_staufD(ax,fd,stau,staufD,nx,ny):
        sampling_fd = np.max([int(len(fd)/ny),1])
        sampling_stau = np.max([int(len(stau)/nx),1])
        data_staufD = block_reduce(np.abs(staufD), block_size=(sampling_fd,sampling_stau), func=np.mean)
        coordinates = np.array([fd,fd])
        coordinates = block_reduce(coordinates, block_size=(1,sampling_fd), func=np.mean, cval=fd[-1])
        data_fd = coordinates[0,:]
        coordinates = np.array([stau,stau])
        coordinates = block_reduce(coordinates, block_size=(1,sampling_stau), func=np.mean, cval=stau[-1])
        data_stau = coordinates[0,:]
        min_nonzero = np.min(data_staufD[np.nonzero(data_staufD)])
        data_staufD[data_staufD == 0] = min_nonzero
        data_staufD = np.log10(data_staufD)
        data_staufD = np.swapaxes(data_staufD,0,1)
        im = ax.pcolormesh(data_fd,data_stau,data_staufD,cmap='viridis',vmin=vmin,vmax=vmax,shading='nearest')
        plt.colorbar(im,ax=ax)

    #set up the canvas
    plot_width = 1000
    plot_height = 700
    plot_dpi = 100
    plot_bottom = 0.15
    plot_top = 0.93
    plot_left = 0.10
    plot_right = 0.95
    plot_wspace = 0.2
    plot_hspace = 0.2
    figure = plt.figure(figsize=(plot_width/plot_dpi,plot_height/plot_dpi),dpi=plot_dpi)
    plt.subplots_adjust(bottom=plot_bottom,top=plot_top,left=plot_left,right=plot_right,wspace=plot_wspace,hspace=plot_hspace)
    ax = figure.add_subplot(1,1,1)
    plot_staufD(ax,fD/1.0e-3,stau/1.0e-3,staufD,500,500)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.set_xlabel(r"$\sqrt{\tau}$ [$\sqrt{\mu s}$]")
    ax.set_ylabel(r"$f_{\rm D}$ [mHz]")
    slider_zeta = mpl.widgets.Slider(plt.axes([0.12,0.02,0.7,0.03]),r'$\zeta$',0.0,zeta_max,valinit=zeta_init)
    slider_err = mpl.widgets.Slider(plt.axes([0.53,0.07,0.4,0.03]),r'$\sigma_{\zeta}$ %',0.0,50.0,valinit=0.0)
    button_save = mpl.widgets.Button(plt.axes([0.12,0.07,0.1,0.03]), "save")
    y_fit = np.linspace(ymin,ymax,num=201,endpoint=True)
    x_u = np.abs(y_fit)*2.*nu0*(slider_zeta.val*(1.+slider_err.val/100.))
    x_d = np.abs(y_fit)*2.*nu0*(slider_zeta.val*(1.-slider_err.val/100.))
    fit_plot_u, = ax.plot(x_u,y_fit,color='red',linestyle='-',markersize=0,alpha=0.5)
    fit_plot_d, = ax.plot(x_d,y_fit,color='red',linestyle='-',markersize=0,alpha=0.5)
    
    def update_zeta(event):
        x_u = np.abs(y_fit)*2.*nu0*(slider_zeta.val*(1.+slider_err.val/100.))
        x_d = np.abs(y_fit)*2.*nu0*(slider_zeta.val*(1.-slider_err.val/100.))
        fit_plot_u.set_xdata(x_u)
        fit_plot_d.set_xdata(x_d)
        figure.canvas.draw_idle()
    def fct_button_save(event):
        plt.close()
    
    slider_zeta.on_changed(update_zeta)
    slider_err.on_changed(update_zeta)
    button_save.on_clicked(fct_button_save)
    plt.show()
    
    return slider_zeta.val,slider_zeta.val*slider_err.val/100.