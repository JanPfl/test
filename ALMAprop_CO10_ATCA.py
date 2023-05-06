import matplotlib.backend_bases
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from matplotlib.ticker import OldScalarFormatter
import numpy as np
from astropy.table import Table
import math
import pylab
import pandas as pd
import matplotlib.patheffects as pe
from pylab import *
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
#from scipy.misc import imresize
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import download_file

image_file = 'hst_mos_0015261_acs_wfc_f814w_sci.fits'
#image_file = download_file(fits_file, cache=True)

# Note that it's better to open the file with a context manager so no
# file handle is accidentally left open.
with fits.open(image_file) as hdus:
    img = hdus[1].data
    wcs = WCS(hdus[1].header)

fig = plt.figure(figsize=(10,6))


# You need to "catch" the axes here so you have access to the transform-function.
ax = fig.add_subplot(111, projection=wcs)
ax.tick_params(axis='y',which='both',direction='in',right=True)
plt.imshow(img, vmin=0.002, vmax=0.01, origin='lower', cmap='gray_r')
plt.xlabel('Right Ascension', fontsize=15, family='serif')
plt.ylabel('Declination', fontsize=15, family='serif', labelpad=-0.6)
plt.minorticks_on()
plt.xticks(fontsize=9, family='serif')
plt.yticks(fontsize=9, family='serif')
# Apply a transform-function:





from astropy.io import ascii
#plt.text(175.240791666667, -26.492, 'DKB12', size=8, color='limegreen', transform=ax.get_transform('world'))


tbl= ascii.read("LAES_PENTERICCI_MAP.dat")
RA=tbl["RA"]
DEC=tbl["DEC"]

plt.scatter(RA,DEC,s=70,marker='2',alpha=1, linewidths=0.5, color='black', label='LAEs Pentericci+00', transform=ax.get_transform('world'))


tbl = ascii.read("KOYAMA_HAE_MAP.dat")
ID=tbl["ID"]
RA=tbl["RA"]
DEC=tbl["DEC"]
plt.scatter(RA,DEC,s=15,marker='o',alpha=1,edgecolor='k',linewidths=1, color='none', label='HAEs Koyama+13', transform=ax.get_transform('world'))

error_kwargs = { "zorder":10}
tbl = ascii.read("DKB_MAP.dat")
ID=tbl["ID"]
RA=tbl["RA"]
DEC=tbl["DEC"]
plt.scatter(RA,DEC,s=85,marker='D',alpha=1,edgecolor='black', linestyle='-', linewidths=1, color='none', label='LABOCA+Herschel Dannerbauer+14', transform=ax.get_transform('world'))



#for ID, RA, DEC in zip(ID, RA, DEC):
#	plt.annotate(ID, xy=(RA, DEC), xycoords=ax.get_transform('world'), size=5, xytext=(1, 1),
#        textcoords='offset points', ha='left', va='bottom')

#plt.text(175.228115833333, -26.4675988888889, '   HAE 1181', transform=ax.get_transform('world'), size=6)
#plt.text(175.227303333333, -26.4732586111111, '   HAE 1162', transform=ax.get_transform('world'), size=6)
#plt.text(175.229934583333, -26.4783413888889, '   HAE 1154', transform=ax.get_transform('world'), size=6)
#plt.text(175.240886666667, -26.5133488888889, '   HAE 1054', transform=ax.get_transform('world'), size=6)
#plt.text(175.244748333333, -26.5062477777778, '   HAE 996', transform=ax.get_transform('world'), size=6)
#plt.text(175.250578333333, -26.4823122222222, '   HAE 779', transform=ax.get_transform('world'), size=6)
#plt.text(175.213590000000, -26.4940769444444, '   HAE 1300', transform=ax.get_transform('world'), size=6)
#plt.text(175.191937500000, -26.4864791666667, '   HAE 902', transform=ax.get_transform('world'), size=6)
#plt.text(175.167067083333, -26.4964027777778, '   HAE 343', transform=ax.get_transform('world'), size=6)
#plt.text(175.155587916667, -26.5048119444444, '   HAE 210', transform=ax.get_transform('world'), size=6)
#plt.text(175.158973750000, -26.5067661111111, '   HAE 255', transform=ax.get_transform('world'), size=6)
#plt.text(175.240173750000, -26.4931488888889, '   HAE 1069', transform=ax.get_transform('world'), size=6)
#plt.text(175.206679166667, -26.4847738888889, '   Spiderweb Galaxy', transform=ax.get_transform('world'), size=8)
#plt.text(175.248409583333, -26.5108577777778, '   HAE 790', transform=ax.get_transform('world'), size=6)
#plt.text(175.249256666667, -26.5118433333333, '   HAE 782', transform=ax.get_transform('world'), size=6)
#plt.text(175.157361666667, -26.4867600000000, '   HAE 229', transform=ax.get_transform('world'), size=6)

tbl = ascii.read("KH_ALMAB7_MAP.dat")
RA=tbl["RA"]
DEC=tbl["DEC"]
plt.scatter(RA,DEC,s=190,marker='s',alpha=1,edgecolor='black',linewidths=1, color='none', label='ALMA Band 7', transform=ax.get_transform('world'))

tbl = ascii.read("MIPS_KOYAMA13_MAP.dat")
RA=tbl["RA"]
DEC=tbl["DEC"]
plt.scatter(RA,DEC,s=45,marker='o',alpha=1,edgecolor='red',linewidths=1, color='none', label='Spitzer/MIPS', transform=ax.get_transform('world'))

tbl = ascii.read("SHUOWEN_ATCA_MAP.dat")
RA=tbl["RA"]
DEC=tbl["DEC"]
plt.scatter(RA,DEC,s=70,marker='o',alpha=1,edgecolor='deepskyblue',linestyle='-', linewidths=1, color='none', label='ATCA CO(1-0) spec-z Jin+21', transform=ax.get_transform('world'))


tbl= ascii.read("KMOS_cat.dat")
RA2=tbl["RA2"]
DEC2=tbl["DEC2"]

plt.scatter(RA2,DEC2,s=140,marker='o',alpha=1,edgecolor='blueviolet', linestyle='--', linewidths=1, color='none', label='VLT/KMOS spec-z', transform=ax.get_transform('world'))


tbl= ascii.read("DRGs.dat")
ID= tbl["ID_S18"]
RA=tbl["RA"]
DEC=tbl["DEC"]
B_ZP=tbl["B_ZP"]
ZP_KS= tbl["ZP_KS"]

df = pd.DataFrame({"ID":ID, "RA":RA, "DEC":DEC, "B_ZP":B_ZP, "ZP_KS":ZP_KS})

df2= df[(df.ZP_KS >= 3) & (df.ZP_KS >= df.B_ZP)]
df3= df[(df.ZP_KS >= 2.5) & (df.ZP_KS <= 3) & (df.ZP_KS >= df.B_ZP)]
df4= df[(df.ZP_KS >= 2.5) & (df.ZP_KS <= df.B_ZP)]


plt.scatter(df2.RA,df2.DEC,s=80,marker='o',alpha=1,edgecolor='black', linestyle='-', linewidths=1, color='orangered', label='DRGs', transform=ax.get_transform('world'))
plt.scatter(df3.RA,df3.DEC,s=80,marker='o',alpha=1,edgecolor='black', linestyle='-', linewidths=1, color='orangered', label=None, transform=ax.get_transform('world'))
plt.scatter(df4.RA,df4.DEC,s=80,marker='o',alpha=1,edgecolor='black', linestyle='-', linewidths=1, color='orangered', label=None, transform=ax.get_transform('world'))



tbl= ascii.read("Tozzi22_spec.dat")
RA2=tbl["RA"]
DEC2=tbl["DEC"]

plt.scatter(RA2,DEC2,s=50,marker='x',alpha=1,edgecolor='black', linestyle='-', linewidth=2, color='gold', label='X-Ray Tozzi+22', transform=ax.get_transform('world'))





tbl= ascii.read("spiderweb_galaxy.dat")
RA2=tbl["RA2"]
DEC2=tbl["DEC2"]

plt.scatter(RA2,DEC2,s=600,marker='+',alpha=1,edgecolor='black', linestyle='-', linewidth=1.5, color='black', label='Spiderweb Galaxy', transform=ax.get_transform('world'))
           

 #,**error_kwargs)
            #,**error_kwargs)
#from PIL import Image
#im = mpimg.imread('contour_dkb03.png')
#print'Original Dimensions : ',img.shape
#imagebox = OffsetImage(im, zoom=1)

#ab = AnnotationBbox(imagebox, xy=(130, 80), xybox=(80., 00.),
#                    xycoords='data',
#                    boxcoords="offset points",
#                    pad=0.5,
#                    arrowprops=dict(
#                        arrowstyle="->",)
#                    )

#ax.add_artist(ab)


############# plot circle fov #############################

from astropy.io import ascii

tbl = ascii.read("Pointing_Hprop.dat")
tbl
import astropy.coordinates as coord
import astropy.units as u


#R_gr
all_rows = tbl[0:35] # get the first (0th) row
ra = coord.Angle(all_rows["RA"], unit=u.hour) # create an Angle object
dec = coord.Angle(all_rows["DEC"], unit=u.degree) # create an Angle object
rgr=all_rows["R_gr"]
RA2=ra.degree
DEC2=dec.degree

#print ra.degree # convert to degrees
#print dec.degree
#print rgr
import matplotlib.patches as patches
import matplotlib as mpl

import itertools as it
#for ras, decl, rgroup in it.izip(RA2, DEC2, rgr):
c = Rectangle((RA2-0.003, DEC2-0.0055), 0.037777, 0.034166, edgecolor='deeppink', facecolor='none', angle=72, linestyle='--', linewidth=4, label=None, transform=ax.get_transform('world'))
d = Rectangle((RA2-0.08, DEC2-0.0125), 0.11, 0.055, edgecolor='green', facecolor='none', angle=0, linestyle='--', linewidth=4, label=None, transform=ax.get_transform('world'))
plt.text(RA2+0.0020, DEC2+0.035, 'HST/WFC3', rotation=20, size=16, color='deeppink', transform=ax.get_transform('world'), weight='bold')
plt.text(RA2+0.00625, DEC2-0.0106, 'ALMA/Band-6 & JWST/NIRCam', size=16, color='green', transform=ax.get_transform('world'), weight='bold')
	#plt.text(ras-0.0495, decl+0.039, 'HST/ACS', rotation=-6, size=16, color='black', transform=ax.get_transform('world'), weight='bold')
        
        #t2 = mpl.transforms.Affine2D().rotate_deg_around(ras-0.03, decl-0.01, -0.5) + ax.transData
        #c.set_transform(t2)
        
ax.add_patch(c)
ax.add_patch(d)
##########################################################





plt.ylim(-200, img.shape[0]-150)
plt.xlim(-100, img.shape[1]+100)

ax.tick_params(direction='in', which='both', length=4, width=1, colors='k',
               grid_color='r', grid_alpha=0.5)
ax.tick_params(axis='y',which='both',direction='in',right=True)
plt.tick_params(axis='both', which='both', direction='in', length=6, width=2, colors='k') #, grid_color='r', grid_alpha=0.5


###############################

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.ops import cascaded_union, unary_union

coords = np.array([
    [175.2280208, -26.4725000],
    [175.1570833, -26.5055833],
    [175.1795833, -26.4889167],
    [175.1839167, -26.4755833],
    [175.2185417, -26.4915917],
    [175.2505833, -26.4823056],
    [175.1670833, -26.4964028],
    [175.1655000, -26.4792222],
    [175.1941667, -26.4863889],
    [175.2012500, -26.4863889],
    [175.2465417, -26.5118611],
    [175.2398333, -26.4933333],
    [175.2599167, -26.4625278]
])

circles = []
for coord in coords:
    circles.append(Point(coord[0], coord[1]).buffer(0.01833))

poly = unary_union(circles).boundary

#fig, ax = plt.subplots()
plt.plot(*poly.xy, color='orange', linewidth=4, label=None, transform=ax.get_transform('world'))


###############################

legend=plt.legend(loc='lower center',framealpha=1,borderaxespad=0.5, facecolor='white', ncol=3, 
                  numpoints=1,scatterpoints = 1, frameon=True, prop={'size':8.7, 'family':'serif'}, labelspacing=0.3) #size was 10 before

legend.legendHandles[0]._sizes = [30]
legend.legendHandles[1]._sizes = [30]
legend.legendHandles[2]._sizes = [30]
legend.legendHandles[3]._sizes = [30]
legend.legendHandles[4]._sizes = [30]
legend.legendHandles[5]._sizes = [30]
legend.legendHandles[6]._sizes = [30]
legend.legendHandles[7]._sizes = [30]
legend.legendHandles[8]._sizes = [30]
#legend.legendHandles[9]._sizes = [30]
legend.legendHandles[9]._sizes = [65]
#legend.legendHandles[10]._sizes = [30]
plt.savefig("ALMAprop_CO10_ATCA.pdf",dpi=150, bbox_inches='tight')


