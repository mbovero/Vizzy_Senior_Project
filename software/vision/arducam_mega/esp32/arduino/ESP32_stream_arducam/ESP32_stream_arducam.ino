#include "Arducam_Mega.h"
#include "ArducamLink.h"

// Define the chip select (CS) pin for the camera
const int CS = 17;

// Create an instance of the Arducam_Mega class
Arducam_Mega myCAM(CS);

ArducamLink myUart;
uint8_t sendFlag = TRUE;
uint32_t readImageLength = 0;
uint8_t jpegHeadFlag     = 0;

// Function to initialize the camera with default settings
void initializeCamera() 
{
  // Initialize the camera
  myCAM.begin();
  //Serial.println("Attempted to start ArduCam Mega...");

  // Register the callback function for image data
  myCAM.registerCallBack(readBuffer, 200, stopPreview);

  // Set default camera settings
  myCAM.takePicture((CAM_IMAGE_MODE)(0x02), (CAM_IMAGE_PIX_FMT)(0x01));   // Set resolution to 640x480 and format to JPG
  myCAM.setBrightness((CAM_BRIGHTNESS_LEVEL)(0x00));                      // Set brightness to default
  myCAM.setContrast((CAM_CONTRAST_LEVEL)(0x00));                          // Set contrast to default
  myCAM.setSaturation((CAM_STAURATION_LEVEL)(0x00));                      // Set saturation to default
  myCAM.setEV((CAM_EV_LEVEL)(0x00));                                      // Set exposure value to default
  myCAM.setAutoWhiteBalanceMode((CAM_WHITE_BALANCE)(0x00));               // Set white balance mode to auto
  myCAM.setColorEffect((CAM_COLOR_FX)(0x00));                             // Set no special effects
  myCAM.setAutoFocus(0x02);  

  Serial.println("Camera initialized with default settings.");
}

// Function to start the preview stream
void startPreviewStream() 
{
  // Start the preview stream
  CamStatus status = myCAM.startPreview((CAM_VIDEO_MODE)(0x02));
  if (status == CAM_ERR_NO_CALLBACK) 
  {
    Serial.println("Error: Callback function not registered!");
    while (1); // Halt if the preview fails to start
  }

  Serial.println("Preview stream started. Sending image data over UART...");
}

uint8_t readBuffer(uint8_t* imagebuf, uint8_t length)
{
    if (imagebuf[0] == 0xff && imagebuf[1] == 0xd8) {
        jpegHeadFlag    = 1;
        readImageLength = 0;
        myUart.arducamUartWrite(0xff);
        myUart.arducamUartWrite(0xAA);
        myUart.arducamUartWrite(0x01);

        myUart.arducamUartWrite((uint8_t)(myCAM.getTotalLength() & 0xff));
        myUart.arducamUartWrite((uint8_t)((myCAM.getTotalLength() >> 8) & 0xff));
        myUart.arducamUartWrite((uint8_t)((myCAM.getTotalLength() >> 16) & 0xff));
        myUart.arducamUartWrite((uint8_t)((myCAM.getTotalLength() >> 24) & 0xff));
        myUart.arducamUartWrite(((CAM_IMAGE_PIX_FMT_JPG & 0x0f) << 4) | 0x01);
    }
    if (jpegHeadFlag == 1) {
        readImageLength += length;
        for (uint8_t i = 0; i < length; i++) {
            myUart.arducamUartWrite(imagebuf[i]);
        }
    }
    if (readImageLength == myCAM.getTotalLength()) {
        jpegHeadFlag = 0;
        myUart.arducamUartWrite(0xff);
        myUart.arducamUartWrite(0xBB);
    }
    return sendFlag;
}

// Stopped preview callback
void stopPreview()
{
  Serial.println("Preview stopped!");
}

// Function to capture and send a single picture
void captureAndSendPicture() 
{
  // Take a picture
  CamStatus status = myCAM.takePicture((CAM_IMAGE_MODE)(0x02), (CAM_IMAGE_PIX_FMT)(0x01));
  if (status != CAM_ERR_SUCCESS) {
    Serial.println("Failed to capture picture!");
    return;
  }

  // Send the picture data over UART
  Serial.println("Sending image data...");
  myUart.cameraGetPicture(&myCAM); // Use the cameraGetPicture function
  Serial.println("Image sent successfully.");
}

// Setup function
void setup() 
{
  // Initialize serial communication
  Serial.begin(921600);
  while (!Serial); // Wait for the serial port to connect
  Serial.println("Connected to serial port!");

  // Initialize the camera
  initializeCamera();

  //Serial.println("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");   // Debug tool

  // Start the preview stream
  startPreviewStream();       // Uncomment for continuous image stream transfer

  //captureAndSendPicture();    // Uncomment for single image transfer upon device reset
}

// Loop function
void loop() 
{
  // Continuously capture and send image data
  myCAM.captureThread();      // Uncomment for continuous image stream transfer
}