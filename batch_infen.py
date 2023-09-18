import os
import time

from tqdm import tqdm

from scripts.infer import acousticbat







def batchintool( proj: str,ppm:str,
        exp: str,
        ckpt: int=None,
        spk: str=None,
        out: str=None,
        title: str=None,
        num: int=1,
        key: int=0,
        gender: float=None,
        seed: int=-1,
        depth: int=-1,
        speedup: int=0,
        mel: bool=False,print_hparams=True,maxppm=None):



    acousticbat( proj=proj,
        exp=exp,
        ckpt=ckpt,
        spk=spk,
        out=out,
        title=title,
        num=num,
        key=key,
        gender=gender,
        seed=seed,
        depth=depth,
        speedup=speedup,
        mel=mel,ppm=ppm,print_hparams=print_hparams,maxppm=maxppm)

if __name__=='__main__':
    cpx='新建文件夹/'
    dsl=os.listdir(cpx)
    ckpt='rc1_unetH'
    ppm='promet/me.wav'
    ppms=ppm.split('/')[-1].replace('.wav','')
    outee=f'boutput/time_{time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())}-ckpt_{ckpt}-ppm_{ppms}'
    # outee = f'boutput/test'
    print_hparams = True

    for idx,i in enumerate(tqdm(dsl)):
        print(f'{idx}:{i}')
        batchintool(proj=f'{cpx}/{i}',exp=ckpt,ppm=ppm,out=outee,print_hparams=print_hparams,maxppm=500
                    # ,speedup=500
                    )
        print_hparams=False

