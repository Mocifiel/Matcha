import numpy as np


def shuffle_mel_by_frame(mel, phone_id, duration, brphone, silphone):
    newmelinputmodel = mel
    startidx = 2
    endidx = len(phone_id) - 1
    if startidx < endidx:
        if np.sum(duration[startidx: endidx]) > 50:
            midstartidx = np.sum(duration[:startidx])
            midendidx = midstartidx + np.sum(duration[startidx: endidx])
            middat = mel[midstartidx: midendidx]
            startdat = mel[:np.sum(duration[:startidx])]
            enddat = mel[-np.sum(duration[-1]):]

            newmel = middat
            newduration = duration[startidx: endidx]
            newphone_id = phone_id[startidx: endidx]
            newphone_idraw = newphone_id
            newdurationraw = newduration

            newdurationacc = np.cumsum(newduration)
            newdurationacc = np.concatenate([[0], newdurationacc], 0)

            newphoneid_melmap = {}
            # map phone index with its duration
            for idx in range(len(newphone_id)):
                newphoneid_melmap[idx] = newmel[newdurationacc[idx]: newdurationacc[idx + 1]]

            # split by br phone
            segtmp = []
            newphseg = []
            idx = 0
            while idx < len(newphone_id):
                # br phones
                if newphone_id[idx] in brphone:
                    segtmp.append(newphone_id[idx])
                    if len(segtmp) > 0:
                        newphseg.append((idx + 1 - len(segtmp), idx + 1))
                    segtmp = []
                else:
                    segtmp.append(newphone_id[idx])
                idx += 1
            if len(segtmp) > 0:
                newphseg.append((idx - len(segtmp), idx))
            rand_seg_ids = np.arange(len(newphseg))
            np.random.shuffle(rand_seg_ids)

            newmelrand = []
            for seg_idx in rand_seg_ids:
                if len(newmelrand) == 0:
                    newmelrand = newmel[newdurationacc[newphseg[seg_idx][0]]: newdurationacc[newphseg[seg_idx][1]]]
                else:
                    newmelrand = np.concatenate([newmelrand, newmel[newdurationacc[newphseg[seg_idx][0]]: newdurationacc[newphseg[seg_idx][1]]]], 0)

            assert(len(newmel) == len(newmelrand))
            newmel = np.array(newmelrand, dtype=np.float32)

            newmelinputmodel = []
            rawmel_idx = 0
            idx = 0
            for ph in newphone_idraw:
                # silence and break related phones
                issil = ph in silphone
                choose = np.random.rand(1)[0]
                # 30% ratio keep the same frame
                if choose < 0.3 or issil:
                    if len(newmelinputmodel) == 0:
                        newmelinputmodel = newphoneid_melmap[idx]
                        if len(newphoneid_melmap[idx]) > 0:
                            newph = newphoneid_melmap[idx]
                            newmelinputmodel = newph
                    else:
                        if len(newphoneid_melmap[idx]) > 0:
                            newmelinputmodel = np.concatenate([newmelinputmodel, newphoneid_melmap[idx]], 0)
                else:
                    dur = newdurationraw[idx]
                    tmpmel = []
                    while dur > 0:
                        if len(tmpmel) == 0:
                            tmpmel = newmel[rawmel_idx: rawmel_idx + 1]
                            rawmel_idx += 1
                            if rawmel_idx == len(newmel):
                                rawmel_idx = 0
                        else:
                            tmpmel = np.concatenate([tmpmel, newmel[rawmel_idx: rawmel_idx + 1]], 0)
                            rawmel_idx += 1
                            if rawmel_idx == len(newmel):
                                rawmel_idx = 0
                        dur -= 1

                    if len(newmelinputmodel) == 0:
                        newmelinputmodel = tmpmel
                        if len(tmpmel) > 0:
                            newmelinputmodel = tmpmel
                    else:
                        if len(tmpmel) > 0:
                            newmelinputmodel = np.concatenate([newmelinputmodel, tmpmel], 0)
                idx += 1

            if np.sum(duration[-1]) != 0:
                newmelinputmodel = np.concatenate([startdat, newmelinputmodel, enddat], 0).astype(np.float32)
            else:
                newmelinputmodel = np.concatenate([startdat, newmelinputmodel], 0).astype(np.float32)

    return newmelinputmodel
