import json, os
import os.path as osp
import math
import numpy as np
import random
import pandas as pd
import cv2
from tqdm import tqdm

Train_list = ["bZsfeA9uRk7.basis", "g8Xrdbe9fir.basis", "4wCTuaUNWEd.basis", "NtnvZSMK3en.basis", "S7uMvxjBVZq.basis",
 "QN2dRqwd84J.basis", "kXAEFtUBNFZ.basis", "E64sjs3Dyfd.basis", "bvdRzJBgJyg.basis", "oStKKWkQ1id.basis", "hbr5fTQhAAa.basis", 
 "jAzZDvf6i67.basis", "AUkcTmUs8mw.basis", "mDPCxA7W1WN.basis", "A9yB3w3UxXV.basis", "BcZUZQ9t4Fe.basis", "uvveHdpUZis.basis",
 "NBuk4gePdJm.basis", "kHvNo8x6Qoe.basis", "12e6joG2B2T.basis", "VoVGtfYrpuQ.basis", "GtM3JtRvvvR.basis", "pVnwDTdMD3h.basis",
 "hTTnuAeSN6d.basis", "fK2vEV32Lag.basis", "iKFn6fzyRqs.basis", "Ty7djLDxQu3.basis", "XRHpoTZjtj7.basis", "kEL17iFsVbw.basis", 
 "rK4jPRTUw15.basis", "TBdN234fGEb.basis", "sjH1uaR68XQ.basis", "pmZMJfFd3Jy.basis", "xAHnY3QzFUN.basis", "uXh7Yr7x12L.basis",
 "ENiCjXWB6aQ.basis", "aKwZ7VJDAjr.basis", "jG9pucmJVBZ.basis", "w8GiikYuFRk.basis", "y3dZtLZCUvN.basis", "HjxjHvpdeoM.basis", 
 "LVgQNuK8vtv.basis", "9Ckja165ren.basis", "a3JCmxobR99.basis", "5jLhtVmWd5F.basis", "VYnUX657cVo.basis", "GZRLndzSrdn.basis", 
 "MVVzj944atG.basis", "bXu6SSWkJY8.basis", "2tqbn5VLQoq.basis", "t8wCA6Qe8uT.basis", "S3YyrKoJ7k6.basis", "1K7P6ZQS4VM.basis", 
 "AwL2QGztLwV.basis", "2dZ1Jivh5if.basis", "g7hUFVNac26.basis", "ENhuWpDE5EB.basis", "yogvKWUrdnw.basis", "PuFu1zFVc4k.basis", 
 "ixTj1aTMup2.basis", "ypcVfePF8TG.basis", "rihBP3nC6p4.basis", "BCWtGkh8CHv.basis", "nvjM2xMma91.basis", "duthTPisf28.basis", 
 "mkvHBa3mEEk.basis", "Xfhi9GYbhqD.basis", "CthA7sQNTPK.basis", "NPHxDe6VeCc.basis", "k7mcxHG65Wh.basis", "kAMF2R7PCqX.basis", 
 "CtZLhCbWFm7.basis", "r77mpaAYUEc.basis", "Pmv1pdeirDT.basis", "eip6PNoeCPr.basis", "N7YVmJQ8sAu.basis", "1SedVoP7zLu.basis", 
 "qWb4MVxqCW7.basis", "AeBfp3hTadB.basis", "q9CAdKfvar2.basis", "DACaFbApXUe.basis", "tzaZQQmUVXZ.basis", "tjs8mFdJ7YN.basis", 
 "tvvDpjzFJGe.basis", "tEafuWwhhwr.basis", "vjMVcmpC1hC.basis", "URjpCob8MGw.basis", "k9UfRPqLm3j.basis", "xc2kFoo9nbw.basis", 
 "wPLokgvCnuk.basis", "DqJKU7YU7dA.basis", "yX5efd48dLf.basis", "d6bYiL1d9Fh.basis", "dQrLTxHvLXU.basis", "ZhXWtW1gd6c.basis", 
 "kGVKV9k9DCT.basis", "qQgcM8T4hiD.basis", "JSgMy8tTACD.basis", "iigzG1rtanx.basis", "k17yptqNRAn.basis", "S3BfyR31Wc9.basis", 
 "s7kPJndncRy.basis", "WnvnMQh4eEa.basis", "92vYG1q49FY.basis", "jGdNyKqGZJw.basis", "SSbSnMigayt.basis", "gQ3xxshDiCz.basis", 
 "Qkm4CooNoPi.basis", "ANmWrL7Kz7h.basis", "gjhYih4upQ9.basis", "CKbwkKufMWM.basis", "NGyoyh91xXJ.basis", "kjUg7BaQF1C.basis", 
 "BLhX6Do8f1t.basis", "AdNTcRg3THp.basis", "58fkJMgLopt.basis", "ZKqtodH1qpa.basis", "udze1CSof5C.basis", "GGBvSFddQgs.basis", 
 "We1N7vBtyGm.basis", "JjsvQEqRxGS.basis", "8QtyGUUtacf.basis", "CQWES1bawee.basis", "5graSmdK3Bj.basis", "L5QEsaVqwrY.basis", 
 "P3kZKzwnEbM.basis", "WpAGGyZFqQj.basis", "janiYDpzM9j.basis", "LqsTKpxKVP2.basis", "3z5dc2yzyCb.basis", "csHQLLFPE3g.basis", 
 "TgWKHxhJAng.basis", "T48WJA2vtru.basis", "fQHGxvurx9L.basis", "rmDFTEWfNcz.basis", "sX9xad6ULKc.basis", "yTgw14aa5ha.basis", 
 "rzzVnFnBLtg.basis", "j2EJhFEQGCL.basis", "w3ZK3Wxvidz.basis", "FgXPKxNp5kK.basis", "uXU4dVyyvWa.basis", "ggNAcMh8JPT.basis", 
 "q6tn1ZjSsG4.basis", "PM558qFsyi8.basis", "SA759ShSs44.basis", "S9M7ybC5ZHu.basis", "zUG6FL9TYeR.basis", "bB6nKqfsb1z.basis", 
 "RHdkyzXFp1k.basis", "UrFKpVJpvHi.basis", "n8AnEznQQpv.basis", "NtVbfPCkBFy.basis", "1sPp3Wz8TCB.basis", "rxGLNxH6eoJ.basis", 
 "t6tH2pNA9X5.basis", "UVdNNRcVyV1.basis", "3goH1WRaCYC.basis", "e3YKRHQRPNe.basis", "u3zrj4Nojev.basis", "suQdyWFG8g9.basis", 
 "PYHRaSotJNh.basis", "ki6Cu76pWzF.basis", "r8b8sRuxdt2.basis", "erfqyP5V4u6.basis", "oQPVc6vwgaq.basis", "CnU5RD6PB3E.basis", 
 "3iZkJUc7KhX.basis", "UZ5rYGiwQgW.basis", "pEeGyoYCEa1.basis", "njMGKG4iwRK.basis", "drvaU627cQh.basis", "KHhgcNqsc9h.basis", 
 "mt5PhMmTk5m.basis", "fxbzYAGkrtm.basis", "nzuiinFMXvf.basis", "JUANmB8jduD.basis", "Hsk3jDzNySy.basis", "MVStfLiYYQv.basis", 
 "hWDDQnSDMXb.basis", "panm7DRsmDn.basis", "WPzCkWEorzk.basis", "1k479icNeHW.basis", "XvJjCZv6SYp.basis", "deNrXzuSss5.basis", 
 "ooq3SnvC79d.basis", "HxmXPBbFCkH.basis", "tj8ngv3woJ3.basis", "U8F9SkAsqbJ.basis", "kQVPtf7ACRw.basis", "EQSguCqe5Rk.basis", 
 "zepmXAdrpjR.basis", "NpCFg9NdUgL.basis", "H7bBanejcc6.basis", "MHYu4LWb6qP.basis", "EN7GiDgxdQ2.basis", "ikChNYDHtRf.basis", 
 "mj5sKX44BmS.basis", "qz3829g1Lzf.basis", "WT4QWwXrMzs.basis", "FnDDfrBZPhh.basis", "z9w4aD7JsiQ.basis", "7GvCP12M9fi.basis", 
 "DZsJKHoqEYg.basis", "NfkadBDgBJV.basis", "16tymPtM7uS.basis", "wuKkTq5GJbi.basis", "DS3nSAaa3Nr.basis", "1hovphK64XQ.basis", 
 "4h4JxvG3cip.basis", "gxttMtT5ZGK.basis", "3Ao63EY7J83.basis", "5RtSdesLuHt.basis", "RHrFkyC59tf.basis", "pAjDzi9kWjE.basis", 
 "1zDbEdygBeW.basis", "u9LiqMn6kA6.basis", "FYYpmNC4gAd.basis", "ochRmQAHtkF.basis", "fJ1GEE6PdHD.basis", "APXAdV48nKT.basis", 
 "5vUupbRRdyH.basis", "XokRUNE3gB1.basis", "WhNyDTnd9g5.basis", "JPMDv7zL4bF.basis", "67ADtrTrBK2.basis", "mQFC1yx29MM.basis", 
 "cFqWyQ4Y9hT.basis", "Ze6tkhg7Wvc.basis", "315QujoX279.basis", "kuxpR2xHUBa.basis", "QKGMrurUVbk.basis", "1Rg1SS1dRpG.basis", 
 "y3K5dmhuukt.basis", "DBBESbk4Y3k.basis", "LGGnLUDPz37.basis", "c6TFyURFrL4.basis", "KAzjXJvZtR3.basis", "erXNfWVjqZ8.basis", 
 "6acdNdTjNbr.basis", "kdw2Uapns3b.basis", "yqNxxJnA3iL.basis", "KjZrPggnHm8.basis", "6AGcGQf2wof.basis", "sCCThjhioJC.basis", 
 "TzQLNfWugiZ.basis", "u9rPN5cHWBg.basis", "LPwS1aEGXBb.basis", "4L4peQsMgfR.basis", "wCqnXzoru3X.basis", "2Pc8W48bu21.basis", 
 "9h5JJxM6E5S.basis", "5m6t1y5EvsT.basis", "GCEb4nmNi7j.basis", "pMntW4YkvvB.basis", "H77MhktmmAF.basis", "SAZ4gvMfxm1.basis", 
 "C8VQQtzUoqV.basis", "osQy15y8EVT.basis", "sNikFfBW8zM.basis", "dTzYwo8Hppu.basis", "Z2DQddYp1fn.basis", "vPoFkhqsJaf.basis", 
 "DoSbsoo4EAg.basis", "LQy8D2nmZ4x.basis", "nMeXfQU4PMS.basis", "QFjJExB2jgE.basis", "aTf5zsbjZMb.basis", "enfahKs8XHw.basis", 
 "dKySjDYsya1.basis", "VhissfC8ggN.basis", "Jfyvj3xn2aJ.basis", "R9fYpvCUkV7.basis", "6HRFAUDqpTb.basis", "nACV8wLu1u5.basis", 
 "NBg5UqG3di3.basis", "EqZacbtdApE.basis","HLBJGGyicLV.basis", "zhzot8MvSjF.basis", "H8rQCnvBgo6.basis", "ECStCRoCNWM.basis", 
 "4GfZ9TTZUwL.basis", "PjnDyQJJ3eM.basis", "GsQBY83r3hb.basis", "TNx8nti6GNi.basis", "RTV2n6fXB2w.basis", "Yr35Q49vqwV.basis", 
 "FxCkHAfgh7A.basis", "XfUxBGTFQQb.basis", "D8bT1ambLFc.basis", "b3CuYvwpzZv.basis", "YY8rqV6L6rf.basis", "UYrgg12a7QN.basis", 
 "nS8T59Aw3sf.basis", "1sM6KvYg3J5.basis", "bMvM1KL4WsR.basis", "cWfRoQnzNiM.basis", "m49MsVC7BwA.basis", "gamLwhSzHci.basis", 
 "otTm4oTrHvc.basis", "aqx6EomMgTf.basis", "kxWk8ZMDE1N.basis", "qvNra81N8BU.basis", "wrq3kiEU4VR.basis", "6imZUJGRUq4.basis", 
 "qWP3MMQM3eJ.basis", "YmEfzspXX5h.basis", "VqCaAuuoeWk.basis", "VSxVP19Cdyw.basis", "p4ZPcGtk6Ex.basis", "VZy9kKQJcUF.basis", 
 "WeyCwVzL53K.basis", "GtqoUWABJ11.basis", "XiJhRLvpKpX.basis", "qSom26FpYzR.basis", "oahi4u45xMf.basis", "KJxdMPgweZG.basis", 
 "8wJuSPJ9FXG.basis", "UQuchpekHRJ.basis", "3Y14etT7365.basis", "R6Byftz8wRN.basis", "x1pTUWx9DPr.basis", "fRZhp6vWGw7.basis", 
 "g7sKCMRfgUS.basis", "XKqDR74W1JU.basis", "yHLr6bvWsVm.basis", "GCCrNuhZ9WY.basis", "yQESfVcg18k.basis", "AENiMBDjVFb.basis", 
 "reHtN7VMWkg.basis", "s19Uyn7AWwv.basis", "uHnM1oqv2JL.basis", "GcfUJ79xCZc.basis", "PB8eQHRTRRK.basis", "kDgLKdMd5X8.basis", 
 "1wypxmRjuUR.basis", "5737gQA9p2T.basis", "ZVPMj4YoZtK.basis", "PFLHHbjscNN.basis", "v46TaF2rxHK.basis", "9hJwm8k7Gka.basis", 
 "8LLjiNrWzJ9.basis", "asrq4PFvdvF.basis", "VWczCD1Hbus.basis", "GNGYKt8XrjF.basis", "rrjjmoZhZCo.basis", "C2hbxeJWmvX.basis", 
 "ZB8o8rMmPdB.basis", "wtaQdtXzYtD.basis", "J5SkB2o1ckv.basis", "o4tckGBtaxz.basis", "Am8wmPcBmtN.basis", "28FFMGySc6D.basis", 
 "RamZzGBBPbT.basis", "EgzWe8N3jZG.basis", "iePHCSf119p.basis", "P3hmFK6Ejnf.basis", "31DHHWieDMS.basis", "mscxX4KEBcB.basis", 
 "XNiSi1YgPRR.basis", "LcAd9dhvVwh.basis", "5K2dTSVihN7.basis", "6YtDG3FhNvx.basis", "eZrc5fLTCmi.basis", "N8oi63yAP2b.basis", 
 "T22dejNjHK7.basis", "QDtpZSqaeyW.basis", "KCvzhHEhdwB.basis", "DsEJeNPcZtE.basis", "NieWWMV6tE4.basis", "krsjseyn6fd.basis", 
 "adddVdvEXUK.basis", "frThKkhTwFT.basis", "vDfkYo5VqEQ.basis", "BEuB32yj7Fb.basis", "6BReaxZUoMg.basis", "pcpn6mFqFCg.basis", 
 "sWAdxSLVPQC.basis", "8nSVqHLMRwQ.basis", "6r9GfBG7u1g.basis", "j2DKmTV5TPV.basis", "Fgtk7tL8R9Y.basis", "MfkErJj6CHF.basis", 
 "XxbS57Z6PDU.basis", "2NwLiyeKcrK.basis", "qAwGYe2GoZp.basis", "A1jHexSJuAW.basis", "n2Tt2eJdqnT.basis", "KcHdFEzySGq.basis", 
 "W16Bm4ysK8v.basis", "qk9eeNeR4vw.basis", "TGVJHgmMGzi.basis", "iTm2PKHUcTJ.basis", "M8PMwoYQTUV.basis", "N17ddiDvJr9.basis", 
 "4RuxhXRmb3V.basis", "AiY2reLtjYJ.basis", "FXuXGH9YQTW.basis", "SSwbmq72C21.basis", "j2Nms3h9XJv.basis", "kfPV7w3FaU5.basis", 
 "3XYAD64HpDr.basis", "gmLDom6XSo7.basis", "Y4L8fjz2yH7.basis", "6HMiy15cxis.basis", "7UdY7HiDnUi.basis", "nHHVbEyHX3t.basis", 
 "3UDjdrwcqMb.basis", "AfKhsVmG8L4.basis", "oz1yTAGPXkh.basis", "iLDo95ZbDJq.basis", "H1D2FZ8TAv1.basis", "iQPq34e8hJX.basis", 
 "4J8N2Ah1a6o.basis", "SBHLgvFTVMZ.basis", "PXAfUkZGMdU.basis", "WEDXu8bWRkq.basis", "yA3RqPqMrGE.basis", "oEPjPNSPmzL.basis", 
 "bDTsgcSK5Qr.basis", "aJg466zMSNt.basis", "wcJYziD5pmF.basis", "dDyovSFuViJ.basis", "NkvRYHk72vA.basis", "FgswoxWb3uN.basis", 
 "3YSDRj9kTU7.basis", "TQSiMZJawkS.basis", "kZhZfAhdnNN.basis", "zJ3fVx3BZYR.basis","kA2nG18hCAr.basis", "mHXUEKEV6gR.basis", 
 "RrfVebebfWf.basis", "YV9M9gZG3YJ.basis", "c9DeZf2fcDf.basis", "Umx6CdjZfvy.basis", "qZ4B7U6XE5Y.basis", "UQ5EhY5wve1.basis", 
 "3PiKdwyfEkX.basis", "q33GehreMrX.basis", "cVppJowrUqs.basis", "nicaPonCxvC.basis", "7PZPFHR3oJc.basis", "wwX4MFiTTrt.basis", 
 "qmvPLqLAgvC.basis", "ZxkSUELrWtQ.basis", "D5dEbkUphhr.basis", "DwDDvGo9QdA.basis", "adgwjGh4NQK.basis", "Nf3aGQTDAA1.basis", 
 "NBWrHFXBF5p.basis", "AuGMayXVFkc.basis", "HsYeeztxPG6.basis", "NRsmXFcVTbN.basis", "uhkqDVMtEnn.basis", "LLecyBe5Eq2.basis", 
 "NEVASPhcrxR.basis", "j2eqyxdYAFW.basis", "uc8QkFS11Hj.basis", "PE6kVEtrxtj.basis", "tJ7XVoEN82a.basis", "UfhK7KNBg5u.basis", 
 "Vnb6uKtzQCU.basis", "MPPDV4Gvybr.basis", "5uXtMs57HmZ.basis", "GfF9TQ34x37.basis", "k3ohRuM6bso.basis", "PaQrTquNd2v.basis", 
 "hXHUtviUKBu.basis", "SrHVAbHUpUX.basis", "gyK27yu7CP4.basis", "GTV2Y73Sn5t.basis", "yX54kr5c5g9.basis", "T7nCRmufFNR.basis", 
 "GMwtBqNLGBs.basis", "5biL7VEkByM.basis", "dNASL765WSN.basis", "v7DzfFFEpsD.basis", "wxixLWuvLjd.basis", "zR6kPe1PsyS.basis", 
 "q28T9C3q2dv.basis", "3CBBjsNkhqW.basis", "9K1WbyTZ456.basis", "UAGeBzZJgkU.basis", "kCHmLFfMDuE.basis", "dcd823nTKH9.basis", 
 "9DnDAhJ7qcj.basis", "mWqBmEyXcXN.basis", "iNpfPhK1sRz.basis", "8B43pG641ff.basis", "SrBPiU6LKxL.basis", "WZDzPCybQvS.basis", 
 "gmuS7Wgsbrx.basis", "TiWanpmC63V.basis", "CxxHb5C8ZsP.basis", "XYyR54sxe6b.basis", "giViJCyCH2C.basis", "LViDMxZp4ZN.basis", 
 "jTTGECZYKRA.basis", "XNoaAZwsWKk.basis", "BqLwEyiLbza.basis", "JY8e73x9ubE.basis", "Wo6kuutE9i7.basis", "2XVvKEDd54w.basis", 
 "6ySDHVkso9e.basis", "37c5w29pYm3.basis", "Lva3QmSMsTr.basis", "D8aaq3PH6dG.basis", "cHumXFzhHUR.basis", "4vwGX7U38Ux.basis", 
 "JWWJBQWHv64.basis", "H81QMurNRM8.basis", "aYhkzj2fEhP.basis", "sLwz8nKD3wF.basis", "YM4nG4pSAEJ.basis", "nW7z5USWzWo.basis", 
 "Y6WjWkVEUks.basis", "o94q92w5PK5.basis", "741Fdj7NLF9.basis", "JiHGQpwKUvd.basis", "nJTPfwbAj4S.basis", "oXzJVhUhmYe.basis", 
 "D2PqRE5ZvyQ.basis", "xGnehmjiCSA.basis", "QKfBMSSy7Hy.basis", "LPEMkRVudUm.basis", "F1Vhvu3osn6.basis", "TZ2jsvNG2nt.basis", 
 "SgkmkWjjmDJ.basis", "WpVxtsP4xxA.basis", "PUNuHY5M7MS.basis", "LU4A39yR8gc.basis", "qDjhFcNqFPi.basis", "mggziYKSc6S.basis", 
 "U3oQjwTuMX8.basis", "mDdyQ6azhVD.basis", "tYvWp85L81G.basis", "C5RbHBQ76DE.basis", "qnKYFQsjnHf.basis", "1S7LAXRdDqK.basis", 
 "zCMdfYaW9iF.basis", "RYzud5W7ZnC.basis", "UbsJXeCkJBA.basis", "3KZbo846fxq.basis", "TSJmdttd2GV.basis", "b3WpMbPFB6q.basis", 
 "7CXbc73tDRf.basis", "RiwBKy2YdQ7.basis", "1UnKg1rAb8A.basis", "p32JzpQyhPk.basis", "JXdzHne1mRo.basis", "JFgrz9MNz4b.basis", 
 "8EqKbkhqE4R.basis", "oKFJo8jpzRW.basis", "P6ajptD9tRP.basis", "eAUmfFLZDR3.basis", "mHJxL9jnCox.basis", "u5atqC7vRCY.basis", 
 "fKP4sxcoxpL.basis", "NwG7cpZnRZb.basis", "pUneSGJDrvY.basis", "77mMEyxhs44.basis", "rWHyWNc6ZbZ.basis", "p6RF8AUer2e.basis", 
 "DGXRxHddGAW.basis", "qgZhhx1MpTi.basis", "nd6Vw5SHCoy.basis", "MLVm7dZk7dp.basis", "by8SK9u18S8.basis", "kJxT5qssH4H.basis", 
 "kyoZhaD9HuW.basis", "1EiJpeRNEs1.basis", "o1F5JVHc6mb.basis", "ij6Fizhrr6c.basis", "8oSQng53cGV.basis", "xcTV5UHYHFV.basis", 
 "5Kw4nGdqYtS.basis", "h5VYFcePkbn.basis", "bCFcvb4zc3N.basis", "QDvRVeWFCjM.basis", "b2e31HFFizw.basis", "1xGrZPxG1Hz.basis",
 "k7vRbGpz44m.basis", "C3ifY177Ldq.basis", "dioA6agn1cP.basis", "PyZonHqd5gy.basis", "fdfRpE6Cfsr.basis", "Wjvsh6jnVsR.basis", 
 "HeSYRw7eMtG.basis", "JHHVv6QZJMm.basis", "zJEEFaNaRbB.basis", "uFCiZVVks57.basis", "DfUQLukPMPc.basis", "9oGV6Y9nNqB.basis", 
 "vj4rZPfVjBQ.basis", "HkseAnWCgqk.basis", "UuwwmrTsfBN.basis", "bAdy4hKf1a1.basis", "mjvN6RDLsPm.basis", "BJovXQkqbC3.basis", 
 "HZ2iMMBsBQ9.basis", "NjyeoK5BLx3.basis", "KsD3yx9nZCv.basis", "kubNyvKJBUX.basis", "knPYW3fibqY.basis", "E1NrAhMoqvB.basis", 
 "awcRF7AZnJu.basis", "8DDKELpgD99.basis", "6qJyEsZNuey.basis", "DBjEcHFg4oq.basis", "226REUyJh2K.basis", "bEdki9cbHDG.basis", 
 "gGMMut83nsX.basis", "xWvSkKiWQpC.basis", "hkmLgL6jrP8.basis", "gDDiZeyaVc2.basis", "Y4idBN66BqG.basis", "QE33dvXyf1U.basis", 
 "MjzBosUX3WW.basis", "8iCxzGNmp4g.basis", "bHKTDQFJxTw.basis", "HPrcqBkKzuy.basis", "tTcuEfoAQXv.basis", "vKuDtqxh7YQ.basis", 
 "t85xiAt9pao.basis", "wz9FcGhrndc.basis", "eUJx9a4u63E.basis", "boHtwWDWtXh.basis", "pYGGNqSbHp1.basis", "qkkRnWghK8e.basis", 
 "d5faFfQRphr.basis", "FcUgaJv6JHA.basis", "fFx6oC7EVp7.basis", "J9adB1bm54A.basis", "f93a9wrxRjG.basis", "7Xp37y8DpSv.basis", 
 "dD37vuCa4FE.basis", "HneuS3CYDFG.basis", "AqGu9nUng3L.basis", "boJrpwDQtu9.basis", "BW1f54ZNVW6.basis", "as8Y8AYx6yW.basis", 
 "oPj9qMxrDEa.basis", "m17UDpW3tHm.basis", "kdFEfVoT1WE.basis", "P8L1328HrLi.basis", "HfMobPm86Xn.basis", "vCfHLVSyL21.basis", 
 "hmRxh2mmzNC.basis", "ACZZiU6BXLz.basis", "T4G9hTR5WSv.basis", "MyxM6trMBUH.basis", "ytXtEYghmvL.basis", "GPyDUnjwZQy.basis", 
 "w7QyjJ3H9Bp.basis", "NcK5aACg44h.basis", "rPwrKEnR3fk.basis", "7dmR22gwQpH.basis", "W4r5JssudHR.basis", "aNri5Gh1ZTE.basis", 
 "P8XJUpcAUkf.basis", "TYDavTf8oyy.basis", "6nvpJEZ8ox5.basis", "wQN24R38a9N.basis", "PK83UqrXjd3.basis", "CFVBbU9Rsyb.basis", 
 "Mre6deDcPCP.basis", "LpFSk2s7me6.basis", "j6fHrce9pHR.basis", "h6nwVLpAKQz.basis", "d88Sc1udFcZ.basis", "YWy9hV7RfQB.basis", 
 "bwoBmU23M2N.basis", "yCCyNxubYcL.basis", "ZNanfzgCdm3.basis", "KWoYKrff5L4.basis"]

Test_list = ["BfzKZxFShtq.basis", "saBtfCeVoJ4.basis", "t3t9ofFLcFU.basis", "WypGcNbCdsH.basis", "fc7RfUCN5mY.basis",
 "8mXffaQTtmP.basis", "Bnq6SeZGL5b.basis", "QVAA6zecMHu.basis", "RfNGMBdVbAZ.basis", "TziyvKgzdAs.basis", 
 "41FNXLAZZgC.basis", "bdp1XNEdvmW.basis", "FRQ75PjD278.basis", "qpcpnP8TosR.basis", "YHmAkqgwe2p.basis", 
 "YJDUB7hWg9h.basis", "z9VLaZqCsW5.basis", "9SpHCfHaNiG.basis", "AMEM2eWycTq.basis", "aosjAwX5Lnq.basis", 
 "aRKASs4e8j1.basis", "DNWbUAJYsPy.basis", "EU6QPFpqdoU.basis", "RaYrxWt5pR1.basis", "SQqGpSHzfSr.basis", 
 "uzH9yHazm9t.basis", "YmWinf3mhb5.basis", "zmZvNTCxMZE.basis", "Coer9RdivP7.basis", "33ypawbKCQf.basis", 
 "4MRLu1yET6a.basis", "8uSpPmctPXC.basis", "C6JvMamYTRg.basis", "F5j7ZLfMm1n.basis", "RcuYAHzrjK7.basis", 
 "y4YiUQwvWGH.basis", "bxwHR9ipFG8.basis", "ceJTwFNjqCt.basis", "cjLuWviyDEo.basis", "rBmEe6ab5VP.basis", 
 "S3r45BMWy6H.basis", "tL6i2PtktSh.basis", "WRphMcFxfhe.basis", "x4LVLSsYWcV.basis", "YGc1h9nNrJP.basis", 
 "4dbCzNN5L5t.basis", "5Poh4Qz68hd.basis", "6TPCFES8fhh.basis", "C7xtw9uhYFn.basis", "F8PSGjTiv61.basis", 
 "sfbj7jspYWj.basis", "VaEwVD182FS.basis", "X6Pct1msZv5.basis", "X9fRPGxw1jS.basis", "Y8Y6ukxGMvn.basis", 
 "ZVScmfktNQ1.basis", "ASKXmHbw68X.basis", "W9YAR9qcuvN.basis", "zWydhyFhvcj.basis", "1mCzDx3EMom.basis", 
 "6EMViBCA2N7.basis", "8qbZhbTc1wX.basis", "99ML7CGPqsQ.basis", "aCtdWA5n56Z.basis", "BUFVGDCQNGb.basis", 
 "CETmJJqkhcK.basis", "qxwfVS8MQ67.basis", "tAQTHnJ7n72.basis", "tpxKD3awofe.basis", "UAByLdpaokx.basis", 
 "w5YEujJKsiy.basis", "XYQdAu1qsK9.basis", "yPKGKBCyYx8.basis", "z8SrvZ4eyqV.basis", "ZwnLFNzxASM.basis", 
 "6vJMULqvYe8.basis", "r38SGhq8aJr.basis", "VKmpsujnc5t.basis", "xccdSFAEPau.basis", "Xuky7E5df6A.basis"]

def extract_regions_from_stitched_image(image_path):
    # Load the stitched image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None

    # Coordinates of the left and right regions in the image (adjust according to your needs)
    # Left image region coordinates
    left_x_start, left_y_start = 81, 180
    left_x_end, left_y_end = 304, 304

    # Right image region coordinates
    right_x_start, right_y_start = 352, 180
    right_x_end, right_y_end = 574, 304

    # Extract regions
    left_image = image[left_y_start:left_y_end, left_x_start:left_x_end]
    right_image = image[right_y_start:right_y_end, right_x_start:right_x_end]

    return left_image, right_image

def extract_green_mask(image):
    if image is None:
        return None

    # Convert to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Define the HSV range for green
    lower_green_hsv = np.array([35, 50, 50])
    upper_green_hsv = np.array([85, 255, 255])

    # Extract green mask
    green_mask = cv2.inRange(hsv_image, lower_green_hsv, upper_green_hsv)

    return green_mask

def compute_list(cam_dir, rel_mat, pose_file, positives, negatives, floor, threshold = 0.01, max_len = 30, resolution=(224,224)):
    rel_matrix = np.load(rel_mat)
    selected_pairs_path = osp.join(cam_dir, "GroundTruth.csv")
    covariant_dir = "/".join(cam_dir.split("/")[:-1])
    covariant_dir = osp.join(covariant_dir,"covariant_images")
    selected_pairs_df = pd.read_csv(selected_pairs_path)

    poses = np.load(pose_file)
    depth_file = osp.join(cam_dir, "saved_dep.npy")
    
    rgb_paths = []
    ref_rgb_paths_all = []
    query_indice = []
    pos_rgb_paths_all = []
    neg_rgb_paths_all = []
    pos_pose_all = []
    neg_pose_all = []
    pos_depth_file_all = []
    neg_depth_file_all = []
    pos_indices_all = []
    neg_indices_all = []
    ref_covariants = []
    pos_covariants = []
    neg_covariants = []
    total_positive_len = 0
    os.makedirs("HM3D", exist_ok=True) 

    for i in range(rel_matrix.shape[0]):
        rgb_paths.append(osp.join(cam_dir,"best_color_"+str(i)+'.png'))

    for i in range(rel_matrix.shape[0]):
        pos_indices = []
        neg_indices = []
        
        for j in range(len(selected_pairs_df)):
            image1_path = selected_pairs_df.iloc[j]["image_1"]
            image2_path = selected_pairs_df.iloc[j]["image_2"]
            idx1 = int(image1_path.split("/")[-1].split(".")[0].split("_")[-1])
            idx2 = int(image2_path.split("/")[-1].split(".")[0].split("_")[-1])
            label = int(selected_pairs_df.iloc[j]["label"])
            if label == 1 and idx1==i:
                pos_indices.append(idx2)
            elif label == 0 and idx1==i:
                neg_indices.append(idx2)
        total_positive_len += len(pos_indices)
        assert(len(pos_indices) < max_len)
        assert(len(neg_indices) < 2*max_len)
        if len(pos_indices) == 0:
            continue
        if len(neg_indices) == 0:
            continue
        if len(pos_indices) >=10:
            pos_indices = pos_indices[:10]
        if len(neg_indices) >=10:
            neg_indices = neg_indices[:10]
        
        for j in range(len(pos_indices)):   
            pos_indices_copy = pos_indices.copy()
            neg_indices_copy = neg_indices.copy()  
            pos_indices_copy[0], pos_indices_copy[j] = pos_indices_copy[j], pos_indices_copy[0]
            random.shuffle(neg_indices_copy)
            # pos_indices_copy += random.choices(pos_indices, k=max_len - len(pos_indices))
            # neg_indices_copy += random.choices(neg_indices, k=2*max_len - len(neg_indices))

            ref_masks = np.zeros(resolution, dtype=np.uint8)
            #### positives #####
            rgb_paths_per = np.array(rgb_paths)[pos_indices_copy].tolist()
            poses_per = np.array(poses)[pos_indices_copy].tolist()
            pos_rgb_paths_all.append(rgb_paths_per)
            pos_indices_all.append(pos_indices_copy)
            pos_covariant_paths = []
            for pos_index in pos_indices_copy:
                scene_seed = rgb_paths_per[0].split("/")[6]
                scene = scene_seed.split("-")[0]
                covariant_path = osp.join(covariant_dir, scene+"_depth_"+str(pos_index)+"_color_"+str(i)+".png")
                assert os.path.exists(covariant_path), f"File does not exist: {covariant_path}"
                if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy"):
                    left_region, right_region = extract_regions_from_stitched_image(covariant_path)
                    assert(left_region is not None and right_region is not None)
                    ref_mask = extract_green_mask(left_region)
                    src_mask = extract_green_mask(right_region)
                    image_width, image_height = resolution
                    ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    ref_masks = np.logical_or(ref_masks, ref_mask)           
                    os.makedirs(f"HM3D/{scene_seed}/{floor}", exist_ok=True) 
                    bool_arr = src_mask == 255 
                    np.save(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy", bool_arr) 
                pos_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy")
            pos_depth_file_all.append(depth_file)
            pos_pose_all.append(poses_per)
            pos_covariants.append(pos_covariant_paths)
            #### negatives #####
            neg_covariant_paths = []
            for neg_index in neg_indices_copy:
                scene = rgb_paths_per[0].split("/")[6].split("-")[0]
                covariant_path = osp.join(covariant_dir, scene+"_depth_"+str(neg_index)+"_color_"+str(i)+".png")
                assert os.path.exists(covariant_path), f"File does not exist: {covariant_path}"
                if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy"):
                    left_region, right_region = extract_regions_from_stitched_image(covariant_path)
                    assert(left_region is not None and right_region is not None)
                    ref_mask = extract_green_mask(left_region)
                    src_mask = extract_green_mask(right_region)
                    image_width, image_height = resolution
                    ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    ref_masks = np.logical_or(ref_masks, ref_mask)
                    os.makedirs(f"HM3D/{scene_seed}/{floor}", exist_ok=True) 
                    bool_arr = src_mask == 255 
                    np.save(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy", bool_arr) 
                neg_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy")
            rgb_paths_per = np.array(rgb_paths)[neg_indices_copy].tolist()
            poses_per = np.array(poses)[neg_indices_copy].tolist()
            neg_rgb_paths_all.append(rgb_paths_per)
            neg_indices_all.append(neg_indices_copy)
            neg_depth_file_all.append(depth_file)
            neg_pose_all.append(poses_per)
            neg_covariants.append(neg_covariant_paths)  
            #### reference #####
            ref_covariant_paths = []
            ref_rgb_paths_all.append(rgb_paths[i])
            query_indice.append(i)
            # if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy"):
            #     np.save(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy", ref_masks) 
            ref_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy")
            ref_covariants.append(ref_covariant_paths)

        for j in range(len(neg_indices)):   
            pos_indices_copy = pos_indices.copy()
            neg_indices_copy = neg_indices.copy()  
            try:
                neg_indices_copy[0], neg_indices_copy[j] = neg_indices_copy[j], neg_indices_copy[0]
            except:
                random.shuffle(neg_indices_copy)

            random.shuffle(pos_indices_copy)
            # pos_indices_copy += random.choices(pos_indices, k=max_len - len(pos_indices))
            # neg_indices_copy += random.choices(neg_indices, k=2*max_len - len(neg_indices))
            
            #### positives #####
            pos_covariant_paths = []
            for pos_index in pos_indices_copy:
                scene = rgb_paths_per[0].split("/")[6].split("-")[0]
                covariant_path = osp.join(covariant_dir, scene+"_depth_"+str(pos_index)+"_color_"+str(i)+".png")
                assert os.path.exists(covariant_path), f"File does not exist: {covariant_path}"
                if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy"):
                    left_region, right_region = extract_regions_from_stitched_image(covariant_path)
                    assert(left_region is not None and right_region is not None)
                    ref_mask = extract_green_mask(left_region)
                    src_mask = extract_green_mask(right_region)
                    image_width, image_height = resolution
                    ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    ref_masks = np.logical_or(ref_masks, ref_mask)
                    os.makedirs(f"HM3D/{scene_seed}/{floor}", exist_ok=True) 
                    bool_arr = src_mask == 255 
                    np.save(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy", bool_arr) 
                pos_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(pos_index)}_{str(i)}.npy")

            rgb_paths_per = np.array(rgb_paths)[pos_indices_copy].tolist()
            poses_per = np.array(poses)[pos_indices_copy].tolist()
            pos_rgb_paths_all.append(rgb_paths_per)
            pos_indices_all.append(pos_indices_copy)
            pos_depth_file_all.append(depth_file)
            pos_pose_all.append(poses_per)
            pos_covariants.append(pos_covariant_paths)
            #### negatives #####
            neg_covariant_paths = []
            for neg_index in neg_indices_copy:
                scene = rgb_paths_per[0].split("/")[6].split("-")[0]
                covariant_path = osp.join(covariant_dir, scene+"_depth_"+str(neg_index)+"_color_"+str(i)+".png")
                assert os.path.exists(covariant_path), f"File does not exist: {covariant_path}"
                if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy"):
                    left_region, right_region = extract_regions_from_stitched_image(covariant_path)
                    assert(left_region is not None and right_region is not None)
                    ref_mask = extract_green_mask(left_region)
                    src_mask = extract_green_mask(right_region)
                    image_width, image_height = resolution
                    ref_mask = cv2.resize(ref_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    src_mask = cv2.resize(src_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                    ref_masks = np.logical_or(ref_masks, ref_mask)
                    bool_arr = src_mask == 255 
                    np.save(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy", bool_arr) 
                neg_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(neg_index)}_{str(i)}.npy")
            rgb_paths_per = np.array(rgb_paths)[neg_indices_copy].tolist()
            poses_per = np.array(poses)[neg_indices_copy].tolist()
            neg_rgb_paths_all.append(rgb_paths_per)
            neg_indices_all.append(neg_indices_copy)
            neg_depth_file_all.append(depth_file)
            neg_pose_all.append(poses_per)
            neg_covariants.append(neg_covariant_paths)  
            #### reference #####
            ref_rgb_paths_all.append(rgb_paths[i])
            query_indice.append(i)
            ref_covariant_paths = []
            ref_covariant_paths.append(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy")
            ref_covariants.append(ref_covariant_paths)
        if not os.path.exists(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy"):
            np.save(f"HM3D/{scene_seed}/{floor}/{str(i)}_{str(i)}.npy", ref_masks) 
    assert(len(query_indice)==len(ref_covariants)==len(pos_indices_all))
    return query_indice, ref_rgb_paths_all, ref_covariants, pos_indices_all, pos_rgb_paths_all, pos_depth_file_all, pos_pose_all, pos_covariants, neg_indices_all, neg_rgb_paths_all, neg_depth_file_all, neg_pose_all, neg_covariants, total_positive_len

def create_dataset(dataset_size, scene_name, mode, data_root, Train_Test_list):
    dataset = []
    total_positive_len = 0
    for i in tqdm(range(dataset_size), desc="Processing"):
        scene = osp.join(data_root, Train_Test_list[i])
        contents = os.listdir(scene)
        floor_folders = [name for name in contents if name.isdigit() and os.path.isdir(os.path.join(scene, name))]
        for floor in floor_folders:
            cam_dir = osp.join(scene, floor, "saved_obs")
            pose = osp.join(cam_dir, "saved_pose.npy")
            rel_mat = osp.join(cam_dir, "rel_mat.npy")

            selected_pairs_path = osp.join(cam_dir, "GroundTruth.csv")
            selected_pairs_df = pd.read_csv(selected_pairs_path)
            positive_indices = []
            negative_indices = []

            query_indice, rgb_paths, ref_covariants, pos_indices, pos_rgb_paths, pos_depth_file, pos_poses, pos_covariants, neg_indices, neg_rgb_paths, neg_depth_file, neg_poses, neg_covariants, len_positives = compute_list(cam_dir, rel_mat, pose, positive_indices, negative_indices, floor)
            total_positive_len += len_positives
            # import pdb; pdb.set_trace()
            # for f in os.listdir(cam_dir):
            #     if f.startswith('best_color_'):
            #         filename = f[:-4].split("_")[-1]
            #         depth_file = "best_depth_"+filename
            #         compute_list(root, floor, f[:-4], depth_file, rel_mat, pose, out_dir)
            for t, indice in enumerate(pos_indices):
                if len(neg_indices[t])==0 or len(pos_indices[t])==0:
                    continue
                data_per = {
                    'scene_name': scene_name,
                    'ref_indice': query_indice[t],
                    'ref_covariants': ref_covariants[t],
                    'pos_indices': pos_indices[t],
                    'ref_rgb_paths': rgb_paths[t],
                    'pos_rgb_list': pos_rgb_paths[t],  # List of image paths
                    'pos_depth_file': pos_depth_file[t],  # List of depth image paths
                    'pos_poses': pos_poses[t],  # List of GT paths
                    'pos_covariants': pos_covariants[t],
                    'neg_indices': neg_indices[t],
                    'neg_rgb_list': neg_rgb_paths[t],  # List of image paths
                    'neg_depth_file': neg_depth_file[t],  # List of depth image paths
                    'neg_poses': neg_poses[t],  # List of GT paths
                    'neg_covariants': neg_covariants[t],
                    'intrinsic_raw': intrinsic_matrix.tolist()  # 4x4 list of lists
                }
                dataset.append(data_per)
    print("total_positive_len:::",total_positive_len)
    return dataset

def save_dataset(dataset, filename):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)

# Example usage
db_root = "../data/hvgg/parta"
data_root = osp.join(db_root,"temp","More_vis")
print('>> Listing all sequences')
sequences = [f for f in os.listdir(data_root)]

set1 = set (sequences)           ##### there are few extra scenes around 18 that nobel removed from the previous implementation of chao. These scenes I could not find in our huggingface dataset.
set2 = set(Train_list + Test_list)
Train_list = list(set(Train_list) - set2.difference(set1))
Test_list = list(set(Test_list) - set2.difference(set1))

train_len = len(Train_list)
test_len = len(Test_list)
assert(len(sequences)==train_len+test_len)
scene_name = "HM3D"
hfov=90 * (math.pi/180)
intrinsic_matrix = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0., 1, 0],
        [0., 0., 0, 1]])

dataset_train = create_dataset(train_len, scene_name, "train", data_root, Train_list)
print("len(dataset_train)::::", len(dataset_train))
save_dataset(dataset_train, "HM3D_train/dataset_train.json")

dataset_test = create_dataset(test_len, scene_name, "test", data_root, Test_list)
print("len(dataset_test)::::", len(dataset_test))
save_dataset(dataset_test, "HM3D_test/dataset_test.json")