/*
 * Navis Robot - ESP32 Motor Controller
 * Raspberry Pi (Master) -> ESP32 (Slave) via Serial
 * 
 * Communication Protocol:
 * Receives: "LEFT,RIGHT\n" (e.g., "150,-100\n")
 * LEFT/RIGHT range: -255 to 255
 * Positive = Forward, Negative = Backward, 0 = Stop
 */

// ===== MOTOR PIN DEFINITIONS =====
// Left Motor
#define L_FWD_PIN 32  // RPWM (Right PWM)
#define L_BWD_PIN 33  // LPWM (Left PWM)

// Right Motor
#define R_FWD_PIN 25  // RPWM
#define R_BWD_PIN 26  // LPWM

// ===== PWM SETTINGS =====
#define PWM_FREQ 20000  // 20 kHz PWM frequency
#define PWM_RES  8      // 8-bit resolution (0-255)

// PWM Channels (ESP32 has 16 PWM channels)
#define L_FWD_CHANNEL 0
#define L_BWD_CHANNEL 1
#define R_FWD_CHANNEL 2
#define R_BWD_CHANNEL 3

// ===== SPEED SETTINGS =====
#define BSPEED    55   // Base Speed (0-255)
#define SPEED     85   // Normal Speed
#define TURN_SPD  100  // Turning Speed

// ===== GLOBAL VARIABLES =====
String inputBuffer = "";
bool commandReady = false;

void setup() {
  // Initialize Serial Communication
  Serial.begin(115200);
  Serial.println("ESP32 Motor Controller Starting...");
  
  // Configure PWM channels
  ledcSetup(L_FWD_CHANNEL, PWM_FREQ, PWM_RES);
  ledcSetup(L_BWD_CHANNEL, PWM_FREQ, PWM_RES);
  ledcSetup(R_FWD_CHANNEL, PWM_FREQ, PWM_RES);
  ledcSetup(R_BWD_CHANNEL, PWM_FREQ, PWM_RES);
  
  // Attach PWM channels to pins
  ledcAttachPin(L_FWD_PIN, L_FWD_CHANNEL);
  ledcAttachPin(L_BWD_PIN, L_BWD_CHANNEL);
  ledcAttachPin(R_FWD_PIN, R_FWD_CHANNEL);
  ledcAttachPin(R_BWD_PIN, R_BWD_CHANNEL);
  
  // Initialize all motors to stop
  stopMotors();
  
  Serial.println("✅ ESP32 Ready!");
  Serial.println("Waiting for commands from Raspberry Pi...");
  Serial.println("Format: LEFT,RIGHT (e.g., 150,-100)");
}

void loop() {
  // Read serial data
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n') {
      // Command complete
      commandReady = true;
    } else {
      inputBuffer += c;
    }
  }
  
  // Process command when ready
  if (commandReady) {
    processCommand(inputBuffer);
    inputBuffer = "";
    commandReady = false;
  }
}

void processCommand(String cmd) {
  // Parse "LEFT,RIGHT" format
  int commaIndex = cmd.indexOf(',');
  
  if (commaIndex == -1) {
    Serial.println("❌ Invalid format. Expected: LEFT,RIGHT");
    return;
  }
  
  // Extract left and right values
  String leftStr = cmd.substring(0, commaIndex);
  String rightStr = cmd.substring(commaIndex + 1);
  
  int leftSpeed = leftStr.toInt();
  int rightSpeed = rightStr.toInt();
  
  // Constrain values to -255 to 255
  leftSpeed = constrain(leftSpeed, -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);
  
  // Apply motor speeds
  setMotorSpeed(leftSpeed, rightSpeed);
  
  // Debug output
  Serial.print("✅ L:");
  Serial.print(leftSpeed);
  Serial.print(" R:");
  Serial.println(rightSpeed);
}

void setMotorSpeed(int left, int right) {
  // LEFT MOTOR
  if (left > 0) {
    // Forward
    ledcWrite(L_FWD_CHANNEL, left);
    ledcWrite(L_BWD_CHANNEL, 0);
  } else if (left < 0) {
    // Backward
    ledcWrite(L_FWD_CHANNEL, 0);
    ledcWrite(L_BWD_CHANNEL, abs(left));
  } else {
    // Stop
    ledcWrite(L_FWD_CHANNEL, 0);
    ledcWrite(L_BWD_CHANNEL, 0);
  }
  
  // RIGHT MOTOR
  if (right > 0) {
    // Forward
    ledcWrite(R_FWD_CHANNEL, right);
    ledcWrite(R_BWD_CHANNEL, 0);
  } else if (right < 0) {
    // Backward
    ledcWrite(R_FWD_CHANNEL, 0);
    ledcWrite(R_BWD_CHANNEL, abs(right));
  } else {
    // Stop
    ledcWrite(R_FWD_CHANNEL, 0);
    ledcWrite(R_BWD_CHANNEL, 0);
  }
}

void stopMotors() {
  ledcWrite(L_FWD_CHANNEL, 0);
  ledcWrite(L_BWD_CHANNEL, 0);
  ledcWrite(R_FWD_CHANNEL, 0);
  ledcWrite(R_BWD_CHANNEL, 0);
}
