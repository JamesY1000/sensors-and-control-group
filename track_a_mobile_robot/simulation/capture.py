"""Optional recording of actual GUI panels and physics states for video replay."""
import json
import subprocess
from types import SimpleNamespace
import numpy as np
import mujoco
from . import evaluation
from .display import SensorDisplay
from ..perception import ArucoLandmarkDetector

class PanelSink:
    def __init__(self,folder):
        self.folder=folder;self.enc=None;self.times=[]
    def submit(self,images,texts):
        # Same panel arrays as the GUI, arranged vertically for editing.
        canvas=np.full((632,320,3),(24,28,34),np.uint8)
        for y,(_,im) in zip((0,282),images):canvas[y:y+im.shape[0],:im.shape[1]]=im
        if self.enc is None:
            self.enc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','320x632','-r','10','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','14','-pix_fmt','yuv420p','-movflags','+faststart',str(self.folder/'gui_sensor_panels.mp4')],stdin=subprocess.PIPE)
        self.enc.stdin.write(canvas.tobytes());self.times.append(self.now)
    def close(self):
        if self.enc:
            self.enc.stdin.close()
            if self.enc.wait():raise RuntimeError('Panel encoding failed')

class CapturedRunLog(evaluation.SimulationRunLog):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.states=[];self.panel=SensorDisplay();self.sink=PanelSink(self.folder);self.panel._publisher=self.sink
        self.window=SimpleNamespace(viewport=mujoco.MjrRect(0,0,1920,1080))
        self.perception=None;self.overlay=(self.folder/'navigation_overlay.jsonl').open('w')
    def tick(self,**kw):
        super().tick(**kw)
        data=kw['data'];backend=kw['backend'];nav=kw['navigator'];now=float(data.time)
        self.states.append((now,data.qpos.copy(),data.qvel.copy()))
        if self.perception is None:
            self.perception=SimpleNamespace(detector=ArucoLandmarkDetector(backend.camera_matrix,backend.dist_coeffs,backend.camera_to_base))
        self.sink.now=now
        self.panel.update(self.window,self.perception,kw['estimate'],kw['output'],nav)
        visual={k:getattr(nav,k) for k in ('trajectory_preview','trajectory_path','coarse_pose','approach_pose','docking_pose','detour_pose','phase')}
        visual['clearance_m']=nav.settings.clearance_m;visual['time']=now
        self.overlay.write(json.dumps(evaluation.json_value(visual),separators=(',',':'))+'\n')
    def close(self,**kw):
        super().close(**kw);self.overlay.close();self.sink.close()
        np.savez_compressed(self.folder/'visual_states.npz',time=np.array([r[0] for r in self.states]),qpos=np.array([r[1] for r in self.states]),qvel=np.array([r[2] for r in self.states]),panel_time=np.array(self.sink.times))
        print('SENSOR CAPTURE SAVED',self.folder,flush=True)
