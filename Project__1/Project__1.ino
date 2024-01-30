#include <Arduino.h>
#include <MeMCore.h>

MeDCMotor motor1(M1);
MeDCMotor motor2(M2);

void moveForward(){
  motor1.run(-75);
  motor2.run(75);
  delay(1000); 
  motor1.run(-150);
  motor2.run(150);
  delay(1000); 
  motor1.run(-75);
  motor2.run(75);
  delay(1000); 
  motor1.stop();
  motor2.stop();
  delay(1000);
  motor1.run(75);
  motor2.run(-75);
  delay(1000); 
  motor1.run(150);
  motor2.run(-150);
  delay(1000); 
   motor1.run(75);
  motor2.run(-75);
  delay(1000); 
  motor1.stop();
  motor2.stop();
}

void setup() {
  pinMode(A7, INPUT);//Start Code when buttonpressed
  while(analogRead(A7) != 0);
  moveForward();
}

void loop() {
  // put your main code here, to run repeatedly:

}
