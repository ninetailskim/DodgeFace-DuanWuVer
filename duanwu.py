import paddlehub as hub
import cv2
import numpy as np
import time
import random
import os
import math
import copy
import glob
from ffpyplayer.player import MediaPlayer
from PIL import Image, ImageDraw, ImageFont
import argparse
import _thread
import asyncio
from playsound import playsound

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

currentTime = 0
lastTime = -1
genTime = [5, 10, 20]
currentIndex = 0
gm = [0, 7, 0]
showimg = None
checkimg = None
minPIXEL = 2500
dangerousPIXEL = 4500
scale = 2
drunk = False
drunkclear = False
level = 5
BM = None
generateFrequency = 3


def thread_playsound(threadName, soundname):
    playsound("resources/music/"+soundname)

def thread_generate(threadName):
    global BM
    global GM
    while GM.lives > 0:
        BM.createBall()
        time.sleep(1.0 / generateFrequency)
        # print(len(BM.balls))
    print("exit thread~" + threadName)

class detUtils():
    def __init__(self):
        super(detUtils, self).__init__()
        self.lastres = None
        self.module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
    
    def distance(self, a, b):
        return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))

    def iou(self, bbox1, bbox2):

        b1left = bbox1['left']
        b1right = bbox1['right']
        b1top = bbox1['top']
        b1bottom = bbox1['bottom']

        b2left = bbox2['left']
        b2right = bbox2['right']
        b2top = bbox2['top']
        b2bottom = bbox2['bottom']

        area1 = (b1bottom - b1top) * (b1right - b1left)
        area2 = (b2bottom - b2top) * (b2right - b2left)

        w = min(b1right, b2right) - max(b1left, b2left)
        h = min(b1bottom, b2bottom) - max(b1top, b2top)

        dis = self.distance([(b1left+b1right)/2, (b1bottom+b1top)/2],[(b2left+b2right)/2, (b2bottom+b2top)/2])

        if w <= 0 or h <= 0:
            return 0, dis
        
        iou = w * h / (area1 + area2 - w * h)
        return iou, dis


    def dodet(self, frame):
        result = self.module.face_detection(images=[frame], use_gpu=False)
        result = result[0]['data']
        if isinstance(result, list):
            if len(result) == 0:
                return None, None
            if len(result) > 1:
                if self.lastres is not None:
                    maxiou = -float('inf')
                    maxi = 0
                    mind = float('inf')
                    mini = 0
                    for index in range(len(result)):
                        tiou, td = self.iou(self.lastres, result[index])
                        if tiou > maxiou:
                            maxi = index
                            maxiou = tiou
                        if td < mind:
                            mind = td
                            mini = index  
                    if tiou == 0:
                        return result[mini], result
                    else:
                        return result[maxi], result
                else:
                    self.lastres = result[0]
                    return result[0], result
            else:
                self.lastres = result[0]
                return result[0], result
        else:
            return None, None

class Resources():
    def __init__(self, xsize, npcsize):
        self.npcsize = npcsize
        self.xsize = xsize
        self.init()

    def init(self):
        zfiles = glob.glob("resources/zongzi/*.png")
        dfiles = glob.glob("resources/wudu/*.png")
        jfiles = glob.glob("resources/xionghuang/*.png")
        xfiles = glob.glob("resources/xiong/*.png")

        zimgs = []
        dimgs = []
        jimgs = []
        ximgs = []

        self.zimgs = [self.pngmergealpha(zf, self.npcsize) for zf in zfiles]
        self.dimgs = [self.pngmergealpha(df, self.npcsize) for df in dfiles]
        self.jimgs = [self.pngmergealpha(jf, self.npcsize) for jf in jfiles]
        self.ximgs = [self.pngmergealpha(xf, self.xsize) for xf in xfiles]

        self.heart = self.pngmergealpha("resources/heart.png", 50)
        self.heartmask = np.zeros_like(self.heart)
        self.heartmask[self.heart > 0] = 1

        self.calculate()

    def getheart(self):
        return self.heart, self.heartmask

    def gethw(self):
        return self.h, self.w

    def getximgs(self, index):
        return self.ximgs[index]

    def getdimgs(self):
        return self.dimgs[random.randint(0, len(self.dimgs)-1)]

    def getzimgs(self):
        return self.zimgs[random.randint(0, len(self.zimgs)-1)]

    def getjimgs(self):
        return self.jimgs[random.randint(0, len(self.jimgs)-1)]

    def gettmusic(self):
        self.tmusic = ["t1.mp3","t2.mp3","t3.mp3","t4.mp3"]
        return self.tmusic[random.randint(0, len(self.tmusic)-1)]

    def getZZmusic(self):
        return "zongzi.mp3"

    def getXHmusic(self):
        return "jiu.mp3"

    def getWDmusic(self):
        return "du.mp3"

    def getopmusic(self):
        return "op.mp3"

    def getop(self):
        return "resources/op.mp4"

    def getwinmusic(self):
        return "win.mp3"

    def getwin(self):
        return "resources/win.mp4"

    def getlosemusic(self):
        return "lose.mp3"

    def getlose(self):
        return "resources/lose.mp4"

    def resize(self, img, size):
        h, w = img.shape[:2]
        ml = max(h, w)
        t = ml / size
        return cv2.resize(img, (int(w / t), int(h / t)))

    def resizeByH(self, img, size):
        h, w = img.shape[:2]
        t = h / size
        return cv2.resize(img, (int(w / t), size))

    def pngmergealpha(self, imgpath, dsize, needmask = False):
        img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        rimg = self.resizeByH(img, dsize)
        mask = rimg[:,:,3]
        mask[mask > 0] = 1
        rimg = rimg[:,:,:-1]
        if needmask:
            mask3 = np.repeat(mask[:,:,np.newaxis], 3, 2)
            return rimg * mask3, mask
        else:
            return rimg * np.repeat(mask[:,:,np.newaxis], 3, 2)

    def calculate(self):
        self.h = 0
        self.w = 0
        tlist = self.dimgs + self.zimgs + self.jimgs
        # tlist.append(self.balloonimg)
        # tlist.append(self.lockimg)
        for img in tlist:
            th, tw = img.shape[:2]
            if th > self.h:
                self.h = th
            if tw > self.w:
                self.w = tw

    def loadMap(self, H, W):
        mapp = cv2.imread("resources/map.png")
        return cv2.resize(mapp, (W, H))

    def PlayVideo(self, video_path, music, H, W):
        video = cv2.VideoCapture(video_path)
        # _thread.start_new_thread(thread_playsound, ("sound1", music))
        lastframe = None
        while True:
            grabbed, frame = video.read()
            if not grabbed:
                break
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            frame = cv2.resize(frame,(W, H))
            # frame = UIM.cv2ImgAddText(showimg, "Score:%d" %(GM.score), int(W/5), int(H/2))
            cv2.imshow("GAME", frame)
            lastframe = frame
        video.release()
        return lastframe

RES = Resources(150, 100)

class GameManager():
    def __init__(self, lives=1):
        super(GameManager, self).__init__()
        self.lives = 1
        self.skill = []
        self.score = 0

    def addPoint(self, num):
        self.score += num * 2 if drunk else num

    def nlive(self):
        self.lives -= 1

    def appendskill(self, skill):
        self.skill.append(skill)

    def play(self):
        if len(self.skill) > 0:
            for s in self.skill:
                if s.finish:
                    self.skill.remove(s)
                else:
                    s.play()

    def reset(self):
        global drunk
        self.lives = 1
        self.skill = []
        self.score = 0
        drunk = False

GM = GameManager()

class UIManager():
    def __init__(self):
        super(UIManager, self).__init__()
        self.heart, self.mask = RES.getheart()

    def cv2ImgAddText(self, img, text, left, top, textColor=(255, 0, 0), textSize=50):
        if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img)

        fontStyle = ImageFont.truetype(
            "font/simsun.ttc", textSize, encoding="utf-8")

        draw.text((left+1, top+1), text, (0, 0, 0), font=fontStyle)
        draw.text((left, top), text, textColor, font=fontStyle)

        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    
    def drawUI(self, canvas, time):
        # draw HP
        startx = 0
        canvas = self.cv2ImgAddText(canvas, "HP: ", startx, 10)
        startx += 75
        # draw heart
        h, w = self.heart.shape[:2]
        if GM.lives == 0:
            canvas = self.cv2ImgAddText(canvas, "0 ", startx, 10)
        else:
            for l in range(GM.lives):
                canvas[10:10+h,startx:startx+w] = canvas[10:10+h,startx:startx+w] * (1-self.mask) + self.mask * self.heart
                startx += (w + 5)
        # draw time icon:
        startx += 15
        canvas = self.cv2ImgAddText(canvas, "Time: %.2f" %time, startx, 10)
        return canvas

UIM = UIManager()

def getPIXEL(x, y, w, h, W, H):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    t = y - h if y - h > 0 else 0
    l = x - w if x - w > 0 else 0
    b = y + h if y + h < H else H - 1
    r = x + w if x + w < W else W - 1
    tt = 0 if y - h > 0 else h - y
    tl = 0 if x - w > 0 else w - x
    tb = 2 * h if y + h < H else h - y + H - 1
    tr = 2 * w if x + w < W else w - x + W - 1
    return int(t), int(l), int(b), int(r), int(tt), int(tl), int(tb), int(tr)

class Skill():
    def __init__(self, interval):
        super().__init__()
        self.stime = 0
        self.interval = interval
        self.finish = False

    def trigger(self):
        self.stime = time.time()
        self.play()

    def play(self):
        pass

class ZongZi(Skill):
    def __init__(self, interval):
        super(ZongZi, self).__init__(interval)

    def play(self):
         if self.finish is False:
            GM.addPoint(100)
            if np.floor(time.time() - self.stime) >= self.interval:
                self.finish = True

class XiongHuang(Skill):
    def __init__(self, interval):
        super(XiongHuang, self).__init__(interval)
        self.init = True

    def play(self):
        global drunk
        global drunkclear
        #print("Lock Play")
        if self.init:
            GM.addPoint(1000)
            drunkclear = True
            self.init = False
        if self.finish is False:
            drunk = True
            if np.floor(time.time() - self.stime) >= self.interval:
                self.finish = True
                drunk = False
        #print("Lock Play end:", llock) 

class FivePoi(Skill):
    def __init__(self, interval):
        super(FivePoi, self).__init__(interval)
    
    def play(self):
        # print("Tansir Play")
        if self.finish is False:
            GM.nlive()
            if np.floor(time.time() - self.stime) >= self.interval:
                self.finish = True

class Ball():
    def __init__(self, x, y, speed_x, speed_y, img, skill, mask=None):
        super().__init__()
        self.x = x
        self.y = y
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.img = img
        if mask is None:
            self.mask = np.zeros_like(img)
            self.mask[img > 0] = 1
        else:
            self.mask = np.repeat(mask[:,:,np.newaxis], 3, 2)
        self.h, self.w = img.shape[:2]
        self.skill = skill
    
    def move(self, W, H):
        global showimg
        global checkimg

        self.x += self.speed_x
        self.y += self.speed_y

        # if self.x > W - self.w/2 or self.x < self.w/2:
        #     self.speed_x = -self.speed_x
        
        # if self.y > H - self.h/2 or self.y < self.h/2:
        #     self.speed_y = -self.speed_y

        if self.y >= H + self.h/2:
            return True
        
        t, l, b, r, tt, tl, tb, tr = getPIXEL(self.x, self.y, self.w/2, self.h/2, W, H)
        
        ctimg = checkimg[t:b, l:r]
        stimg = showimg[t:b, l:r]

        if np.sum(ctimg[self.mask[tt:tb,tl:tr]>0]) > 0:
            self.skill.trigger()
            if self.skill.finish is False:
                GM.appendskill(self.skill)
            if isinstance(self.skill, ZongZi):
                _thread.start_new_thread(thread_playsound, ("sound1",RES.getZZmusic()))
            elif isinstance(self.skill, XiongHuang):
                _thread.start_new_thread(thread_playsound, ("sound1",RES.getXHmusic()))
            else:
                _thread.start_new_thread(thread_playsound, ("sound1",RES.getWDmusic()))
            return True
        else:
            showimg[t:b,l:r] = showimg[t:b,l:r] * (1 - self.mask[tt:tb,tl:tr]) +  self.mask[tt:tb,tl:tr] * self.img[tt:tb,tl:tr]
            return False

class BallManager():
    def __init__(self, level, W, H):
        super(BallManager, self).__init__()
        self.balls = []
        self.startTime = math.floor(time.time())
        self.lastTime = -1
        self.level = level
        self.genPerSec = 5
        self.W = W
        self.H = H
        self.reset()

    def createBall(self):
        h,w = RES.gethw()
        
        speed = self.level if self.level > 5 else 5

        speed_x = 0
        speed_y = random.randint(speed*2, speed*4)

        x, y = self.randomXY(h+20, w+20)

        ratio = random.randint(1, 100)
        if ratio < 20:
            ratio = random.randint(1, 100)
            if ratio < 50: 
                b = Ball(x, y, speed_x, speed_y, RES.getdimgs(), FivePoi(0))
            else:
                b = Ball(x, y, speed_x, speed_y, RES.getjimgs(), XiongHuang(random.randint(1,10)))
        else:
            b = Ball(x, y, speed_x, speed_y, RES.getzimgs(), ZongZi(0))
        self.balls.append(b) 

    def driveBall(self):
        global drunkclear
        # print(len(self.balls))
        for b in self.balls:
            if b.move(self.W, self.H):
                self.balls.remove(b)
        if drunkclear:
            self.balls = []
            drunkclear = False
        if GM.lives <= 0:
            return True
        return False

    def randomXY(self, h, w):
        x = 0
        y = 0
        while x < w/2 or x > self.W-w/2-1:
            x = random.randint(0, self.W-1)
        y = h + 10
        return x, y

    def reset(self):
        self.balls = []
        self.startTime = math.floor(time.time())

    def startGenerate(self):
        _thread.start_new_thread(thread_generate, ("generate",))

def main(args):
    global BM
    global showimg
    global checkimg
    level = args.level

    restart = True
    DU = detUtils()

    videoStream = 0
    cap = cv2.VideoCapture(videoStream)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(W, H)
    if max(H, W) < 1000:
        H, W = H * scale, W * scale
    
    backCanvas = RES.loadMap(H, W)
    # print("backCanvas Shape", backCanvas.shape)
    ret, frame = cap.read()
    # RES.PlayVideo(RES.getop(), RES.getopmusic(), H, W)
    
    
    BM = BallManager(level, W, H)

    while restart:
        restart = False

        GM.reset()
        BM.reset()
        BM.startGenerate()
        startTime = time.time()
        # _thread.start_new_thread(thread_generate, ("generate",))
        
        print("Fuck???")

        # while True:
        #     print(len(BM.balls))

        lastres = None
        timgPlayer = None

        while True:
            ret, frame = cap.read()
            if videoStream == 0:
                frame = cv2.flip(frame, 1)

            cv2.imshow("self", frame)

            if ret == True:
                frame = cv2.resize(frame, (W, H))

                # cv2.imshow("frame", frame)
                # cv2.waitKey(0)

                if drunk:
                    timgPlayer = RES.getximgs(1)
                else:
                    timgPlayer = RES.getximgs(0)
                imgPlayer = copy.deepcopy(timgPlayer)
                h,w = imgPlayer.shape[:2]

                detres, ress = DU.dodet(frame)
                GM.play()

                if detres is not None:
                    lastres = detres
                    showimg = copy.deepcopy(backCanvas)
                    # print(detres)
                    checkimg = np.zeros_like(frame)
                    top = detres['top']
                    right = detres['right']
                    left = detres['left']
                    bottom = detres['bottom']

                    # ttimg = cv2.rectangle(frame,(int(detres['left']), int(detres['top'])),(int(detres['right']), int(detres['bottom'])),(0,0,255),5)
                    # cv2.imshow("test", ttimg)
                    # cv2.waitKey(0)
                    if drunk:
                        cx = W - (right+left)/2
                    else:
                        cx = (right+left)/2
                    t, l, b, r, tt, tl, tb, tr = getPIXEL(cx, (top+bottom)/2, w/2, h/2, W, H)
                    
                    imgPlayer = imgPlayer[tt:tb, tl:tr]
                    maskPlayer = np.zeros_like(imgPlayer)
                    maskPlayer[imgPlayer > 0] = 1
                    
                    showimg[H - h - 10:H - 10, l:r] = showimg[H - h - 10:H - 10, l:r] * (1 - maskPlayer) + imgPlayer
                    showimg = showimg.astype(np.uint8)
                    checkimg[H - h - 10:H - 10, l:r] = checkimg[H - h - 10:H - 10, l:r] + maskPlayer

                    gameover = BM.driveBall()
                    showimg = showimg.astype(np.uint8)

                    # showimg = UIM.drawUI(showimg, 100 - (time.time() - startTime))
                    showimg = UIM.cv2ImgAddText(showimg, "Score:%d Time:%.2f" %(GM.score, 100 - (time.time() - startTime)), 10, 10)
                    if gameover:
                        loseimg = RES.PlayVideo(RES.getlose(), RES.getlosemusic(), H, W)

                        loseimg = UIM.cv2ImgAddText(loseimg, "You Lose! Score:%d " %(GM.score), int(W/5), int(H/2))
                        cv2.imshow('GAME', loseimg)
                        if cv2.waitKey(0) == ord('r'):
                            restart = True
                        break
                    else:
                        cv2.imshow('GAME', showimg)
                        cv2.waitKey(1)
                else:
                    if lastres is None:
                        if showimg is None:
                            showimg = copy.deepcopy(backCanvas)
                        showimg = UIM.cv2ImgAddText(showimg, "没检测到你的脸", int(W/5), int(H/2))
                        cv2.imshow('GAME', showimg)
                        if cv2.waitKey(0) == ord('r'):
                            restart = True
                        break
                    else:
                        detres = lastres
                        showimg = copy.deepcopy(backCanvas)
                        checkimg = np.zeros_like(frame)
                        top = detres['top']
                        right = detres['right']
                        left = detres['left']
                        bottom = detres['bottom']
                        if drunk:
                            cx = W - (right+left)/2
                        else:
                            cx = (right+left)/2
                        t, l, b, r, tt, tl, tb, tr = getPIXEL(cx, (top+bottom)/2, w/2,h/2, W, H)

                        imgPlayer = imgPlayer[tt:tb, tl:tr]
                        maskPlayer = np.zeros_like(imgPlayer)
                        maskPlayer[imgPlayer > 0] = 1

                        showimg[H - h - 10:H - 10, l:r] = showimg[H - h - 10:H - 10, l:r] * (1 - maskPlayer) + imgPlayer
                        showimg = showimg.astype(np.uint8)
                        checkimg[H - h - 10:H - 10, l:r] = checkimg[H - h - 10:H - 10, l:r] + maskPlayer

                        gameover = BM.driveBall()
                        showimg = showimg.astype(np.uint8)

                        showimg = UIM.drawUI(showimg, 100 - (time.time() - startTime))
                        if gameover:
                            loseimg = RES.PlayVideo(RES.getlose(), RES.getlosemusic(), H, W)

                            loseimg = UIM.cv2ImgAddText(loseimg, "You Lose! Score:%d " %(GM.score), int(W/5), int(H/2))
                            cv2.imshow('GAME', loseimg)
                            if cv2.waitKey(0) == ord('r'):
                                restart = True
                            break
                        else:
                            cv2.imshow('GAME', showimg)
                            cv2.waitKey(1)
            else:
                if showimg is None:
                    showimg = copy.deepcopy(backCanvas)
                showimg = UIM.cv2ImgAddText(showimg, "请检查摄像头", int(W/5), int(H/2))
                cv2.imshow('GAME', showimg)
                if cv2.waitKey(0) == ord('r'):
                    restart = True
                break
            
            if time.time() - startTime > 100:
                winimg = RES.PlayVideo(RES.getwin(), RES.getwinmusic(), H, W)
                winimg = UIM.cv2ImgAddText(winimg, "Score:%d" %(GM.score), int(W/5), int(H/2))
                cv2.imshow('GAME', winimg)
                if cv2.waitKey(0) == ord('r'):
                    restart = True
                break
    
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=5)
    args=parser.parse_args()
    main(args)