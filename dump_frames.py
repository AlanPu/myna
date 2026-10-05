import math,struct,subprocess,sys
SR=16000;HOP=0.02
start,dur,lo,hi,step=[float(x) for x in sys.argv[1:6]]
o=subprocess.run(["ffmpeg","-v","error","-ss",str(start),"-t",str(dur),"-i","audio2/native.webm","-f","f32le","-ac","1","-ar",str(SR),"-"],capture_output=True,check=True).stdout
n=len(o)//4; pcm=struct.unpack("<%df"%n,o[:n*4])
w=int(SR*0.025); hop=int(SR*HOP)
env=[math.sqrt(sum(v*v for v in pcm[i:i+w])/w) for i in range(0,len(pcm)-w+1,hop)]
sm=[sum(env[max(0,i-3):min(len(env),i+4)])/len(env[max(0,i-3):min(len(env),i+4)]) for i in range(len(env))]
print("frames %d  %.1f - %.1f"%(len(env),start,start+len(env)*HOP))
for i in range(len(env)):
    t=start+i*HOP
    if lo<=t<=hi and i%step==0:
        db=20*math.log10(sm[i]+1e-12)
        print("  %6.2fs %7.1fdB  %s"%(t,db,"#"*int(max(0,db+55)/2)))
