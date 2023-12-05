#include <Arduino.h>
#include <MeMCore.h>
MeDCMotor motor1(M1);  
MeDCMotor motor2(M2);
MeLineFollower lineFollower(PORT_2);
unsigned long startMillis;  
unsigned long currentMillis;
const unsigned long period = 1000;
const byte ledPin = 13;  


uint8_t speed = 100;



void setup() {
  pinMode(A7, INPUT);//Start Code when buttonpressed
  while(analogRead(A7) != 0);
  Serial.begin(9600);
  startMillis = millis();
}


void loop() {
  currentMillis = millis();
  lineFollow();  
}


void lineFollow(){
  int sensorState = lineFollower.readSensors();


  if(speed>230) {
    speed=230;
  }
  switch(sensorState)
  {
    case S1_IN_S2_IN:
      moveForward();
      break;
    case S1_IN_S2_OUT:
      TurnLeft();
      break;
    case S1_OUT_S2_IN:
      TurnRight();
      break;
    case S1_OUT_S2_OUT:
      if (30000 >= (currentMillis - startMillis)){
        motor1.run(100);
        motor2.run(-150);
      }
      else if (120000 > (currentMillis - startMillis)){
        motor1.run(150);
        motor2.run(-90);
      }
      else if (180000 >= (currentMillis - startMillis)){
        motor1.run(-100);
        motor2.run(-100);
      }
      else if (180000 < (currentMillis - startMillis)){
        motor1.stop();
        motor2.stop();
      }
      break;
  }
}




void moveForward()
{
  motor1.run(-speed);
  motor2.run(speed);
}
void moveBackward()
{
  motor1.run(speed);
  motor2.run(-speed);
}
void TurnLeft()
{
  motor1.run(-50); // Turn left
  motor2.run(speed);
}
void TurnRight()
{
  motor1.run(-speed); // Turn right
  motor2.run(50);
}
void Stop()
{
  motor1.run(0);
  motor2.run(0);
}
