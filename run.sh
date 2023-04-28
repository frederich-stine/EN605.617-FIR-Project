#!/bin/bash

#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of FIR FFT project"
echo "----------------------------------------------------------------"
echo " This runner demonstrates a running the FIR FFT project on three different"
echo " audio files with two different FIR filters. This runner shows the performance"
echo " differences between the CPU and GPU implementations of the algorithms"
echo " This runner does this with two different FFT sizes."
echo ""
echo " You can pause execution at any time and listen to the resulting audio and view"
echo " the resulting FFT files."
echo ""

# ------------- 440Hz 5Sec -------------
read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - 440Hz 5Sec - FFT Size 1024 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_05sec.wav \
    demo_audio/440Hz_44100Hz_16bit_05sec_filtered.wav\
    0 1024

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - 440Hz 5Sec - FFT Size 1024 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_05sec.wav \
    demo_audio/440Hz_44100Hz_16bit_05sec_filtered.wav\
    1 1024

read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - 440Hz 5Sec - FFT Size 4096 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_05sec.wav \
    demo_audio/440Hz_44100Hz_16bit_05sec_filtered.wav\
    0 4096

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - 440Hz 5Sec - FFT Size 4096 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_05sec.wav \
    demo_audio/440Hz_44100Hz_16bit_05sec_filtered.wav\
    1 4096

read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - 440Hz 5Sec - FFT Size 1024 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_05sec.wav \
    demo_audio/440Hz_44100Hz_16bit_05sec_filtered.wav\
    0 1024

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - 440Hz 5Sec - FFT Size 1024 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_05sec.wav \
    demo_audio/440Hz_44100Hz_16bit_05sec_filtered.wav\
    1 1024

read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - 440Hz 5Sec - FFT Size 4096 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_05sec.wav \
    demo_audio/440Hz_44100Hz_16bit_05sec_filtered.wav\
    0 4096

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - 440Hz 5Sec - FFT Size 4096 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_05sec.wav \
    demo_audio/440Hz_44100Hz_16bit_05sec_filtered.wav\
    1 4096


# ------------- 440Hz 30Sec -------------
read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - 440Hz 30Sec - FFT Size 1024 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_30sec.wav \
    demo_audio/440Hz_44100Hz_16bit_30sec_filtered.wav\
    0 1024

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - 440Hz 30Sec - FFT Size 1024 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_30sec.wav \
    demo_audio/440Hz_44100Hz_16bit_30sec_filtered.wav\
    1 1024

read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - 440Hz 30Sec - FFT Size 4096 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_30sec.wav \
    demo_audio/440Hz_44100Hz_16bit_30sec_filtered.wav\
    0 4096

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - 440Hz 30Sec - FFT Size 4096 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_30sec.wav \
    demo_audio/440Hz_44100Hz_16bit_30sec_filtered.wav\
    1 4096

read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - 440Hz 30Sec - FFT Size 1024 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_30sec.wav \
    demo_audio/440Hz_44100Hz_16bit_30sec_filtered.wav\
    0 1024

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - 440Hz 30Sec - FFT Size 1024 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_30sec.wav \
    demo_audio/440Hz_44100Hz_16bit_30sec_filtered.wav\
    1 1024

read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - 440Hz 30Sec - FFT Size 4096 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_30sec.wav \
    demo_audio/440Hz_44100Hz_16bit_30sec_filtered.wav\
    0 4096

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - 440Hz 30Sec - FFT Size 4096 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/440Hz_44100Hz_16bit_30sec.wav \
    demo_audio/440Hz_44100Hz_16bit_30sec_filtered.wav\
    1 4096

# ------------- Cantina Band -------------
read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - Cantina Band 60s - FFT Size 1024 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/CantinaBand60.wav \
    demo_audio/CantinaBand60-Filtered.wav\
    0 1024

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - Cantina Band 60s - FFT Size 1024 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/CantinaBand60.wav \
    demo_audio/CantinaBand60-Filtered.wav\
    1 1024

read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - Cantina Band 60s - FFT Size 4096 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/CantinaBand60.wav \
    demo_audio/CantinaBand60-Filtered.wav\
    0 4096

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - Cantina Band 60s - FFT Size 4096 - bandstop"
./fast-fir demo_filters/bandstop_440hz_44100hz.csv \
    demo_audio/CantinaBand60.wav \
    demo_audio/CantinaBand60-Filtered.wav\
    1 4096

read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - Cantina Band 60s - FFT Size 1024 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/CantinaBand60.wav \
    demo_audio/CantinaBand60-Filtered.wav\
    0 1024

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - Cantina Band 60s - FFT Size 1024 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/CantinaBand60.wav \
    demo_audio/CantinaBand60-Filtered.wav\
    1 1024

read -p "Press enter to continue:"
clear
echo "CPU FIR and FFT - Cantina Band 60s - FFT Size 4096 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/CantinaBand60.wav \
    demo_audio/CantinaBand60-Filtered.wav\
    0 4096

read -p "Press enter to continue:"
clear
echo "GPU FIR and FFT - Cantina Band 60s - FFT Size 4096 - highpass"
./fast-fir demo_filters/highpass_600hz_44100hz.csv \
    demo_audio/CantinaBand60.wav \
    demo_audio/CantinaBand60-Filtered.wav\
    1 4096