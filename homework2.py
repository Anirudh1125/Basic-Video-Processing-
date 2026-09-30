"""CSI 471/571 Homework 2. Run: python homework2.py [--video data/input.mp4]."""
from pathlib import Path
import argparse, json, shutil, subprocess
import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'

def save(path, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8)).save(path)

def normalize(a):
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a) if hi == lo else 255 * (a-lo)/(hi-lo)

def convolve(a, k):
    """True convolution; zero extension. Even kernels use floor(size/2) anchor."""
    h,w=k.shape
    p=np.pad(a, ((h//2,h-1-h//2),(w//2,w-1-w//2)), mode='constant')
    return np.einsum('ijmn,mn->ij', sliding_window_view(p,(h,w)),k[::-1,::-1])

def problem1(img):
    d=OUT/'problem1';d.mkdir(parents=True,exist_ok=True)
    g3=np.array([[1,2,1],[2,4,2],[1,2,1]],float)/16
    g5=np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]],float)/273
    arrays={ 'median':np.median(sliding_window_view(np.pad(img,1),(3,3)),axis=(-2,-1)),
             'gaussian3':convolve(img,g3), 'gaussian5':convolve(img,g5)}
    kernels={ 'roberts':(np.array([[-1,1.]]),np.array([[-1.],[1.]])),
      'prewitt':(np.array([[-1,0,1]]*3,float),np.array([[-1,-1,-1],[0,0,0],[1,1,1]],float)),
      'sobel':(np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float),np.array([[-1,-2,-1],[0,0,0],[1,2,1]],float))}
    stats={}
    for name,(kx,ky) in kernels.items():
        gx,gy=convolve(img,kx),convolve(img,ky)
        arrays[name+'_x']=gx;arrays[name+'_y']=gy
        arrays[name+'_magnitude']=np.hypot(gx,gy)
    arrays['laplacian']=convolve(img,np.array([[0,1,0],[1,-4,1],[0,1,0]],float))
    arrays['dog']=arrays['gaussian5']-arrays['gaussian3']
    for name,a in arrays.items():
        np.save(d/(name+'.npy'),a)
        view=a if name in ['median','gaussian3','gaussian5'] else normalize(a)
        save(d/(name+'.png'),view)
        stats[name]={'min':float(a.min()),'max':float(a.max())}
    save(d/'original.png',img)
    return stats

def problem2():
    """Draw eight shapes by direct NumPy array indexing; no drawing library."""
    img=np.zeros((400,800),np.uint8)
    y,x=np.indices(img.shape)
    for row in range(2):
        cy=100+200*row;filled=row==1
        # Squares: 100 x 100, with an exact one-pixel outline.
        box=(abs(x-100)<=49)&(abs(y-cy)<=49)
        interior=(abs(x-100)<49)&(abs(y-cy)<49)
        img[box if filled else box&~interior]=255
        for cx,rx,ry in [(300,50,50),(500,75,25)]:
            outer=((x-cx)/rx)**2+((y-cy)/ry)**2<=1
            inner=((x-cx)/(rx-1))**2+((y-cy)/(ry-1))**2<1
            img[outer if filled else outer&~inner]=255
        # Triangle vertices (700,cy-50), (650,cy+50), (750,cy+50).
        inside=(y>=cy-50)&(y<=cy+50)&(abs(x-700)<=(y-(cy-50))/2)
        if filled:img[inside]=255
        else:
            core=np.zeros_like(inside)
            core[1:-1,1:-1]=inside[1:-1,1:-1]&inside[:-2,1:-1]&inside[2:,1:-1]&inside[1:-1,:-2]&inside[1:-1,2:]
            img[inside&~core]=255
    save(OUT/'problem2'/'shapes.png',img)
    return img

def haar_layer(a):
    p,q,r,s=a[0::2,0::2],a[0::2,1::2],a[1::2,0::2],a[1::2,1::2]
    return (p+q+r+s)/4,(p+q-r-s)/4,(p-q+r-s)/4,(p-q-r+s)/4

def inverse_layer(a,h,v,d):
    o=np.empty((a.shape[0]*2,a.shape[1]*2),float)
    o[0::2,0::2]=a+h+v+d;o[0::2,1::2]=a+h-v-d
    o[1::2,0::2]=a-h+v-d;o[1::2,1::2]=a-h-v+d
    return o

def problem4(img):
    dest=OUT/'problem4';dest.mkdir(parents=True,exist_ok=True)
    a=img.copy();layers=[];canvas=img.copy();coeff={}
    for level in range(9):
        a,h,v,d=haar_layer(a);layers.append((h,v,d));n=a.shape[0]
        canvas[:n,:n]=a;canvas[:n,n:2*n]=h+128
        canvas[n:2*n,:n]=v+128;canvas[n:2*n,n:2*n]=d+128
        save(dest/f'layer_{level:02d}.png',canvas)
        for label,arr in [('H',h),('V',v),('D',d)]:coeff[f'{label}{level}']=arr
    coeff['Apix']=a
    np.savez(dest/'coefficients.npz',**coeff)
    recon=a.copy()
    for h,v,d in reversed(layers):recon=inverse_layer(recon,h,v,d)
    err=np.abs(img-recon)
    assert np.max(err)==0, 'Haar reconstruction must be exact'
    save(dest/'reconstructed.png',recon);save(dest/'absolute_error.png',err)
    np.save(dest/'reconstructed.npy',recon)
    return {'max_reconstruction_error':float(err.max()),'mean_absolute_error':float(err.mean()),
      'equal_pixels':int(np.count_nonzero(img==recon)),'total_pixels':int(img.size),
      'Apix':float(a[0,0]),'levels':9}

def run(cmd):
    subprocess.run([str(s) for s in cmd],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

def frame_difference(a,b):
    return np.clip(b.astype(np.int16)-a.astype(np.int16)+128,0,255).astype(np.uint8)

def problem3(video):
    if not shutil.which('ffmpeg'):raise RuntimeError('Install FFmpeg and add it to PATH.')
    dest=OUT/'problem3';frames=dest/'frames';diffs=dest/'differences'
    for folder in [frames,diffs]:
        folder.mkdir(parents=True,exist_ok=True)
        for p in folder.glob('*.png'):p.unlink()
    # Canonical RGB source: five seconds, 24 fps, aspect-preserving 320x240 canvas.
    vf='scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2,fps=24'
    run(['ffmpeg','-y','-i',video,'-t','5','-vf',vf,'-an',frames/'frame_%05d.png'])
    files=sorted(frames.glob('frame_*.png'))
    if len(files)<2:raise ValueError('Video must contain at least two frames.')
    for i in range(len(files)-1):
        a=np.array(Image.open(files[i]).convert('RGB'));b=np.array(Image.open(files[i+1]).convert('RGB'))
        save(diffs/f'diff_{i+1:05d}.png',frame_difference(a,b))
    for label,folder,pattern in [('original',frames,'frame_%05d.png'),('difference',diffs,'diff_%05d.png')]:
        for codec in ['msmpeg4v2','ffv1']:
            cmd=['ffmpeg','-y','-framerate','24','-i',folder/pattern,'-an','-c:v',codec]
            cmd+=['-q:v','2','-pix_fmt','yuv420p'] if codec=='msmpeg4v2' else ['-pix_fmt','bgr0']
            run(cmd+[dest/f'{label}_{codec}.avi'])
    # FFV1 must preserve the source RGB frames exactly.
    for label,folder,pattern in [('original',frames,'frame_*.png'),('difference',diffs,'diff_*.png')]:
        raw=subprocess.run(['ffmpeg','-v','error','-i',str(dest/f'{label}_ffv1.avi'),'-f','rawvideo','-pix_fmt','rgb24','-'],check=True,stdout=subprocess.PIPE).stdout
        decoded=np.frombuffer(raw,np.uint8).reshape(-1,240,320,3)
        expected=sorted(folder.glob(pattern));assert len(decoded)==len(expected)
        assert all(np.array_equal(a,np.array(Image.open(p).convert('RGB'))) for a,p in zip(decoded,expected))
    info={'input_frames':len(files),'difference_frames':len(files)-1,'fps':24,'width':320,'height':240,'ffv1_exact_rgb_verified':True,
      'file_sizes_bytes':{p.name:p.stat().st_size for p in dest.glob('*.avi')}}
    (dest/'video_metrics.json').write_text(json.dumps(info,indent=2))
    return info

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--video',type=Path);args=parser.parse_args()
    img=np.array(Image.open(ROOT/'data'/'lenag.png').convert('L'),dtype=np.float64)
    assert img.shape==(512,512)
    metrics={'problem1':problem1(img),'problem4':problem4(img)};problem2()
    if args.video:metrics['problem3']=problem3(args.video.resolve())
    else:metrics['problem3']={'status':'PENDING: student-recorded video required'}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics['problem4'],indent=2))
if __name__=='__main__':main()
