CFLAGS=-c -fPIC -O3

libjam_rms_mgeint.so : jam_axi_rms_mgeint.o 
	$(CC) -shared jam_axi_rms_mgeint.o -o libjam_rms_mgeint.so -lm -L.

jam_axi_rms_mgeint.o : jam_axi_rms_mgeint.c jam.h
	$(CC) $(CFLAGS)  jam_axi_rms_mgeint.c

clean:
	rm *.o 
