"""Plot the liquid-water Magnus approximation and its logarithmic derivative.

Alduchov and Eskridge (1996), equation 21, doi:
10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2.
This is a theoretical illustration, not a fit to the climate-index dataset.

Equation values are calculated directly; no observations are fitted.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]/'Figures'
DATA=HERE.parent/'Data'
STEM='figure_01_thermodynamic_basis'
BLUE,RED,INK='#2166AC','#B2182B','#262626'

def saturation(t):
    return 6.1094*np.exp(17.625*t/(243.04+t))

def sensitivity(t):
    return 100*17.625*243.04/(243.04+t)**2

def plot():
    t=np.linspace(-30,40,701)
    numeric=100*(np.log(saturation(t+1e-4))-np.log(saturation(t-1e-4)))/2e-4
    np.testing.assert_allclose(sensitivity(t),numeric,rtol=1e-8,atol=1e-7)
    assert [round(float(sensitivity(x)),2) for x in [0,10,25]]==[7.25,6.69,5.96]
    # Independent high-precision checks of the coefficients and calculations.
    with localcontext() as ctx:
        ctx.prec=40
        for x in [-40,0,10,25,50]:
            d=Decimal(x)
            expected_pressure=Decimal('6.1094')*(Decimal('17.625')*d/(Decimal('243.04')+d)).exp()
            expected_sensitivity=100*Decimal('17.625')*Decimal('243.04')/(Decimal('243.04')+d)**2
            np.testing.assert_allclose(saturation(x),float(expected_pressure),rtol=1e-14)
            np.testing.assert_allclose(sensitivity(x),float(expected_sensitivity),rtol=1e-14)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,
        'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.8,'axes.grid':True,'grid.alpha':.22,'grid.linewidth':.5,
        'legend.frameon':False,'pdf.fonttype':42,'savefig.dpi':400,'text.color':INK,
        'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.8))
    fig.subplots_adjust(left=.085,right=.98,bottom=.25,top=.88,wspace=.32)
    axes[0].plot(t,saturation(t),color=BLUE,lw=2)
    axes[0].set(title='(a) Saturation vapour pressure',ylabel='Vapour pressure (hPa)',ylim=(0,80),yticks=[0,20,40,60,80])
    axes[1].plot(t,sensitivity(t),color=RED,lw=2)
    axes[1].set(title='(b) Temperature sensitivity',ylabel='Local sensitivity (% K⁻¹)',ylim=(5,10),yticks=[5,6,7,8,9,10])
    axes[1].axhline(7,color='.4',ls='--',lw=1,label='7% K⁻¹ reference')
    axes[1].legend(loc='upper right',fontsize=10)
    for x,offset in [(0,(8,12)),(10,(-18,-23)),(25,(-33,-22))]:
        axes[1].plot(x,sensitivity(x),'o',color=RED,ms=4)
        axes[1].annotate(f'{sensitivity(x):.2f}',(x,sensitivity(x)),xytext=offset,
            textcoords='offset points',fontsize=10,color=INK)
    for ax in axes:
        ax.axvline(0,color='.6',ls=':',lw=.8)
        ax.set(xlim=(-30,40),xticks=np.arange(-30,41,10),xlabel='Temperature (°C)')
    fig.text(.5,.025,'Liquid-water saturation; below 0 °C, the reference phase is supercooled water.',ha='center',fontsize=10)
    for ext in ['png','pdf']:
        fig.savefig(HERE/f'{STEM}.{ext}',facecolor='white')
    plt.close(fig)
