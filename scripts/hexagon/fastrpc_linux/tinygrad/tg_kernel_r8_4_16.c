/* Rendered by tinygrad (DEV=DSP) from capture_tinygrad.py. Do not edit. */
typedef float float16 __attribute__((aligned(64),ext_vector_type(16)));
__attribute__((noinline)) void r_8_4_16(float* restrict __attribute__((align_value(128))) data0_32, float* restrict __attribute__((align_value(128))) data1_128, float* restrict __attribute__((align_value(128))) data2_64) {
  for (int Lidx1 = 0; Lidx1 < 8; Lidx1++) {
    float16 val0 = (*((float16*)((data1_128+(Lidx1<<4)))));
    for (int Lidx2 = 0; Lidx2 < 4; Lidx2++) {
      float val1 = (*(data2_64+(Lidx2+4)));
      float val2 = (*(data2_64+(Lidx2+8)));
      float val3 = (*(data2_64+(Lidx2+12)));
      float val4 = (*(data2_64+(Lidx2+16)));
      float val5 = (*(data2_64+(Lidx2+20)));
      float val6 = (*(data2_64+(Lidx2+24)));
      float val7 = (*(data2_64+(Lidx2+28)));
      float val8 = (*(data2_64+(Lidx2+32)));
      float val9 = (*(data2_64+(Lidx2+36)));
      float val10 = (*(data2_64+(Lidx2+40)));
      float val11 = (*(data2_64+(Lidx2+44)));
      float val12 = (*(data2_64+(Lidx2+48)));
      float val13 = (*(data2_64+(Lidx2+52)));
      float val14 = (*(data2_64+(Lidx2+56)));
      float val15 = (*(data2_64+(Lidx2+60)));
      float val16 = (*(data2_64+Lidx2));
      *(data0_32+((Lidx1<<2)+Lidx2)) = ((val0[0]*val16)+(val0[1]*val1)+(val0[2]*val2)+(val0[3]*val3)+(val0[4]*val4)+(val0[5]*val5)+(val0[6]*val6)+(val0[7]*val7)+(val0[8]*val8)+(val0[9]*val9)+(val0[10]*val10)+(val0[11]*val11)+(val0[12]*val12)+(val0[13]*val13)+(val0[14]*val14)+(val0[15]*val15));
    }
  }
}
