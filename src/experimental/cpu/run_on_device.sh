set -e
set -o xtrace

ninja -f android.build.ninja
adb push build/android/bench /data/local/tmp/bench
adb shell /data/local/tmp/bench
