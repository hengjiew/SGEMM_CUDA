from os import system


BK = [8, 16]
BM = [128, 256]
BN = [128, 256]
TM = [8, 16]
TN = [8, 16]
WM = [32, 64]
WN = [32, 64]
NUM_THREADS = [256]

src = "../src/runner.cu"

for nt in NUM_THREADS:
  for bm in BM:
    for bn in BN:
      for bk in BK:
        for wm in WM:
          for wn in WN:
            for tm in TM:
              for tn in TN:
                # Check if warp size is 32.
                if (wm // tm) * (wn // tn) != 32:
                  continue
                # Check if warp count number of threads.
                if (bm // wm) * (bn // wn) != (nt // 32):
                  continue
                # Check if double buffers fit in shared memory.
                if 2 * (bm + 4 + bn) * bk * 4 > 48 * 1024:
                  continue

                # Print parameters
                print("block  tile: M {:d} N {:d} K {:d}".format(bm, bn, bk))
                print("warp   tile: M {:d} N {:d}".format(wm, wn, bk))
                print("thread tile: M {:d} N {:d}".format(tm, tn))
                print("block layout by warp: {:d} {:d}".format(bm // wm, bn // wn))
                print("warp  layout by thread: {:d} {:d}".format(wm // tm, wn // tn))
                # Edit source code.
                system('sed -i "s/const uint K13_BM = .*/const uint K13_BM = {:d};/g" {}'.format(bm, src))
                system('sed -i "s/const uint K13_BN = .*/const uint K13_BN = {:d};/g" {}'.format(bn, src))
                system('sed -i "s/const uint K13_BK = .*/const uint K13_BK = {:d};/g" {}'.format(bk, src))
                system('sed -i "s/const uint K13_WM = .*/const uint K13_WM = {:d};/g" {}'.format(wm, src))
                system('sed -i "s/const uint K13_WN = .*/const uint K13_WN = {:d};/g" {}'.format(wn, src))
                system('sed -i "s/const uint K13_TM = .*/const uint K13_TM = {:d};/g" {}'.format(tm, src))
                system('sed -i "s/const uint K13_TN = .*/const uint K13_TN = {:d};/g" {}'.format(tn, src))
                system('sed -i "s/const uint K13_NUM_THREADS = .*/const uint K13_NUM_THREADS = {:d};/g" {}'.format(nt, src))
                system('make')
                system('./sgemm 13')
                system ('rm -f sgemm')

