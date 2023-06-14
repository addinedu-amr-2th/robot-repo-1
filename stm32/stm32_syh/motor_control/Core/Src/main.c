/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2023 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stdio.h"
#include "stdlib.h"
#include "motor_encoder.h"
#include <math.h>
#include <string.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
encoder_instance enc_instanceC;

uint8_t g_rx_buf[256] = {0, };
uint8_t g_recv_data[256] = {0, };
uint8_t g_rx_index = 0;
uint8_t data_len = 0;

int16_t g_ma_motor_speed = 0;
int16_t g_mb_motor_speed = 0;
int16_t g_mc_motor_speed = 0;
int16_t g_md_motor_speed = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

uint8_t process_protocol(void);
void send_current_state(void);
void send_resonse_protocol(uint8_t);
uint8_t calc_checksum(uint8_t*, uint8_t);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

int16_t encoder_velocityC;
int32_t encoder_positionC;

uint16_t counterA = 0;
uint16_t counterB = 0;
uint16_t counterC = 0;
uint16_t counterD = 0;
uint16_t directionA = 0;
uint16_t directionB = 0;
uint16_t directionC = 0;
uint16_t directionD = 0;

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart == &huart1)
	{
		HAL_UART_Receive_IT(&huart1, &g_rx_buf[g_rx_index], 1);
		g_rx_index++;
	}
}
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{

	if (htim->Instance == TIM6){

		if(g_ma_motor_speed >= 0)
		{
			HAL_GPIO_WritePin(GPIOD, GPIO_PIN_2, 0);
			HAL_GPIO_WritePin(GPIOC, GPIO_PIN_12, 1);
			__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_4, g_ma_motor_speed);
		}
		else {
			HAL_GPIO_WritePin(GPIOD, GPIO_PIN_2, 1);
			HAL_GPIO_WritePin(GPIOC, GPIO_PIN_12, 0);
			__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_4, -1 * g_ma_motor_speed);
		}

		if(g_mb_motor_speed >= 0)
		{
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4, 0);
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, 1);
			__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_3, g_ma_motor_speed);
		}
		else {
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4, 1);
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, 0);
			__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_3, -1 * g_ma_motor_speed);
		}

		if(g_mc_motor_speed >= 0)
		{
			HAL_GPIO_WritePin(GPIOC, GPIO_PIN_5, 0);
			HAL_GPIO_WritePin(GPIOC, GPIO_PIN_4, 1);
			__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_2, g_ma_motor_speed);
		}
		else {
			HAL_GPIO_WritePin(GPIOC, GPIO_PIN_5, 1);
			HAL_GPIO_WritePin(GPIOC, GPIO_PIN_4, 0);
			__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_2, -1 * g_ma_motor_speed);
		}

		if(g_md_motor_speed >= 0)
		{
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, 0);
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, 1);
			__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_1, g_ma_motor_speed);
		}
		else {
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, 1);
			HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, 0);
			__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_1, -1 * g_ma_motor_speed);
		}

//		counterA = __HAL_TIM_GET_COUNTER(&htim2);
//		counterB = __HAL_TIM_GET_COUNTER(&htim3);
//		counterC = __HAL_TIM_GET_COUNTER(&htim4);
//		counterD = __HAL_TIM_GET_COUNTER(&htim5);
//
//		directionA = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim2);
//		directionB = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim3);
//		directionC = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim4);
//		directionD = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim5);
//
//		update_encoder(&enc_instanceC, &htim4);
//		encoder_positionC = enc_instanceC.position;
//		encoder_velocityC = enc_instanceC.velocity;
	}
}

//void HAL_TIM_IC_CaptureCallback
//void HAL_TIM_PeriodElapsedCallback
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM2_Init();
  MX_TIM3_Init();
  MX_TIM4_Init();
  MX_TIM5_Init();
  MX_TIM8_Init();
  MX_TIM6_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */
  HAL_TIM_Encoder_Start_IT(&htim2, TIM_CHANNEL_ALL);
  HAL_TIM_Encoder_Start_IT(&htim3, TIM_CHANNEL_ALL);
  HAL_TIM_Encoder_Start_IT(&htim4, TIM_CHANNEL_ALL);
  HAL_TIM_Encoder_Start_IT(&htim5, TIM_CHANNEL_ALL);

  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_1);  // start the pwm md
  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_2);  // start the pwm mc
  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_3);  // start the pwm mb
  HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_4);  // start the pwm ma

  HAL_TIM_Base_Start_IT(&htim6);

  __HAL_UART_ENABLE_IT(&huart1,UART_IT_RXNE);
  HAL_UART_Receive_IT(&huart1, &g_rx_buf[g_rx_index], 1);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  while (1)
  {
//	uint8_t len = 0;
//	len = process_protocol();
//
//	if(len != 0)
//	{
//		// response for command
//		uint8_t cmd = g_recv_data[0];
//		uint8_t need_res = g_recv_data[len - 1];
//
//		if(cmd == 0x01)
//		{
//			g_mc_motor_speed = (int16_t)((g_recv_data[1] << 8) | g_recv_data[2]);
//			g_md_motor_speed = (int16_t)((g_recv_data[3] << 8) | g_recv_data[4]);
//			g_ma_motor_speed = (int16_t)((g_recv_data[5] << 8) | g_recv_data[6]);
//			g_mb_motor_speed = (int16_t)((g_recv_data[7] << 8) | g_recv_data[8]);
//
//			if(need_res)
//			{
//				send_resonse_protocol(len);
//			}
//		}
//	}

	//HAL_Delay(10);

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
uint8_t process_protocol()
{
  if(HAL_UART_Receive(&huart1, &g_rx_buf[g_rx_index], 1, 10000) == HAL_OK)
  {
    g_rx_index++;
  }

  if(g_rx_index > 6)
  {
    if(g_rx_buf[g_rx_index - 1] == 0xFD)
    {
      if(g_rx_buf[g_rx_index - 2] == 0xFA)
      {
        uint8_t packet_len = g_rx_buf[g_rx_index - 4];
        // Check Header
        if((g_rx_buf[g_rx_index - packet_len - 5] == 0xFE) && (g_rx_buf[g_rx_index -packet_len - 6] == 0xFA))
        {
          // Check checksum
          uint8_t calc_crc = calc_checksum(&g_rx_buf[g_rx_index - packet_len - 4], packet_len + 1);

          if(calc_crc == g_rx_buf[g_rx_index - 3])
          {
            // Check completed.
            for(int i = 0; i < packet_len; i++)
            {
              g_recv_data[i] = g_rx_buf[g_rx_index - packet_len - 4 + i];
            }

            memset(g_rx_buf, 0, 256);
            g_rx_index = 0;

            return packet_len;
          }
        }
      }
    }
  }

  return 0;
}

//void send_current_state(void)
//{
//  uint8_t send_data[20] = {0, };
//
//  send_data[0] = 0xFA;
//  send_data[1] = 0xFE;
//  send_data[2] = 0x92;
//
//  // int16_t l_state = int16_t(l_current_state * 50);  // enconder for second
//  send_data[3] = (uint8_t)(g_l_current_encoder >> 24);
//  send_data[4] = (uint8_t)(g_l_current_encoder >> 16);
//  send_data[5] = (uint8_t)(g_l_current_encoder >> 8);
//  send_data[6] = (uint8_t)(g_l_current_encoder);
//
//  // int16_t r_state = int16_t(r_current_state * 50);  // enconder for second
//  send_data[7] = (uint8_t)(g_r_current_encoder >> 24);
//  send_data[8] = (uint8_t)(g_r_current_encoder >> 16);
//  send_data[9] = (uint8_t)(g_r_current_encoder >> 8);
//  send_data[10] = (uint8_t)(g_r_current_encoder);
//
//  // int16_t r_state = int16_t(r_current_state * 50);  // enconder for second
//  send_data[11] = (uint8_t)(g_r_current_encoder >> 24);
//  send_data[12] = (uint8_t)(g_r_current_encoder >> 16);
//  send_data[13] = (uint8_t)(g_r_current_encoder >> 8);
//  send_data[14] = (uint8_t)(g_r_current_encoder);
//
//  // int16_t r_state = int16_t(r_current_state * 50);  // enconder for second
//  send_data[15] = (uint8_t)(g_r_current_encoder >> 24);
//  send_data[16] = (uint8_t)(g_r_current_encoder >> 16);
//  send_data[17] = (uint8_t)(g_r_current_encoder >> 8);
//  send_data[18] = (uint8_t)(g_r_current_encoder);
//
//  send_data[19] = 17;
//
//  int sum = 0;
//  for(int i = 0; i < 17; i++)
//  {
//    sum += send_data[3+i];
//  }
//  send_data[20] = (uint8_t)sum;
//
//  send_data[21] = 0xFA;
//  send_data[22] = 0xFD;
//
//  Serial.write(send_data, 20);
//}


void send_resonse_protocol(uint8_t len)
{
  uint8_t send_data[len+6];

  //header
  send_data[0] = 0xFA;
  send_data[1] = 0xFE;

  // cmd + data + response
  for(int i = 0; i < len; i++)
  {
    send_data[2+i] = g_recv_data[i];
  }

  //LEN
  send_data[2+len] = len;

  //CMD에 0x90 더하기 : response를 보내는거니까
  send_data[2] = send_data[2] + 0x90;

  // checksum
  uint16_t sum = 0;
  for(int i = 0; i < 2+len; i++)
  {
    sum += send_data[i];
  }
  send_data[3+len] = (uint8_t)sum;

  // footer
  send_data[4+len] = 0xFA;
  send_data[5+len] = 0xFD;

  HAL_UART_Transmit(&huart1, send_data, len+6 ,10000);
}


uint8_t calc_checksum(uint8_t* data, uint8_t len)
{
  uint16_t sum = 0;
  for(int i = 0; i < len; i++)
  {
    sum += data[i];
  }

  return (uint8_t)sum;
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */