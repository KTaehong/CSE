#include <Arduino.h>
#include <MeMCore.h>

MeDCMotor motor1(M1);
MeDCMotor motor2(M2);
MeRGBLed led(0,30);


void stop(int y){
  led.setColor(255,0,0);
  motor1.stop();
  motor2.stop();
  led.show();
  delay(y);
}

void moveForward(int x){
  //1.6 second of 100 speed moves mbot 12 inches
  delay(500);
  motor1.run(-97);
  motor2.run(100);
  led.setColor(0,255,0);
  led.show();
  delay(133*x); 
  stop(0);
  
}

void turnLeft(){
  led.setColor(255,0,0);
  delay(500);
  motor1.run(100);
  motor2.run(100);  
  led.show();
  delay(490);
  motor1.stop();
  motor2.stop();
  led.show();
  delay(2460);
}

void turnRight(){
  led.setColor(255,0,0);
  delay(500);
  motor1.run(-100);
  motor2.run(-100);  
  delay(700);
  led.show();
  motor1.stop();
  motor2.stop();
  led.show();
  delay(2465);
}


void setup() {
  pinMode(A7, INPUT);//Start Code when buttonpressed
  while(analogRead(A7) != 0);
  led.setpin(13);
  turnRight();  
  
}

void loop() {
  // put your main code here, to run repeatedly:

}
