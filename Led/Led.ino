#include <Arduino.h>
#include <MeMCore.h>

MeRGBLed led(0,30);
int x, y, z;
void setup() {
  
  // put your setup code here, to run once:
  led.setpin(13);
}

void loop() {
  x = random(0,255);
  y = random(0,255);
  z = random(0,255);
  // put your main code here, to run repeatedly:
  led.setColorAt(0,x,y,z);//right side
  led.setColorAt(1,0,0,0);//left side
  led.show();
  delay(100);
  buzzerOn():
  delay(1000);
  buzzerOff();
  delay(1000);
}
