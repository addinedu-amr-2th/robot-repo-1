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
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stdio.h"
#include "stdlib.h"
#include "motor_encoder.h"
#include <math.h>
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
encoder_instance enc_instanceA;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

float_t encoder_velocityA;
float_t encoder_positionA;

uint16_t counterA = 0;
uint16_t counterB = 0;
uint16_t counterC = 0;
uint16_t counterD = 0;
uint16_t directionA = 0;
uint16_t directionB = 0;
uint16_t directionC = 0;
uint16_t directionD = 0;


//void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
//{
//	counter = __HAL_TIM_GET_COUNTER(&htim4);
//
//	update_encoder(&enc_instance, &htim4);
//	encoder_position = enc_instance.position;
//	encoder_velocity = enc_instance.velocity;
//}

//void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
//{
//	if (htim->Instance == TIM6){
//		counterC = __HAL_TIM_GET_COUNTER(&htim4);
//		//countC = (short)counterC;
//		update_encoder(&enc_instance, &htim4);
//		encoder_position = enc_instance.position;
//		encoder_velocity = enc_instance.velocity;
//	}
//}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{

	if (htim->Instance == TIM6){


		counterA = __HAL_TIM_GET_COUNTER(&htim2);
		counterB = __HAL_TIM_GET_COUNTER(&htim3);
		counterC = __HAL_TIM_GET_COUNTER(&htim4);
		counterD = __HAL_TIM_GET_COUNTER(&htim5);

		directionA = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim2);
		directionB = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim3);
		directionC = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim4);
		directionD = __HAL_TIM_IS_TIM_COUNTING_DOWN(&htim5);

		update_encoder(&enc_instanceA, &htim2);
		encoder_positionA = enc_instanceA.position;
		encoder_velocityA = enc_instanceA.velocity;
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
  //HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_1);  // start the pwm1
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  float_t fre[500];
  // uint16_t period[500];
  float_t len = 500.0;
  float_t fre_max = 500.0;
  float_t fre_min = 0.0;
  float_t flexible = 4;

  float_t deno;
  float_t melo;
  float_t delt = fre_max - fre_min;

  float_t timer_freq = 1000000.0;

  while (1)
  {
	  	// MD - 방향
	  	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, 1);
	  	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, 0);
	  	// MC + 방향
	  	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_5, 0);
	  	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_4, 1);
	  	// MB - 방향
	  	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4, 1);
	  	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_5, 0);
	  	// MA + 방향
	  	HAL_GPIO_WritePin(GPIOD, GPIO_PIN_2, 0);
	  	HAL_GPIO_WritePin(GPIOC, GPIO_PIN_12, 1);

	  	for(int i=0; i<len; i++)
	  	{
	  		melo = flexible * (i - len/2) / (len/2);
	  		deno = 1.0 / (1+expf(-melo));
	  		fre[i] = delt * deno + fre_min;
	  		// period[i] = (uint16_t)(timer_freq/fre[i]);

//	  		__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_1, (int)fre[i]);
//	  		__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_2, (int)fre[i]);
//	  		__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_3, (int)fre[i]);
	  		__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_4, (int)fre[i]);
	  		HAL_Delay(1);
	  	}

	  	HAL_Delay(1000-1);

	  	for(int i=len-1; i>=0; i--)
	  	{
//	  		__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_1, (int)fre[i]);
//	  		__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_2, (int)fre[i]);
//	  		__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_3, (int)fre[i]);
	  		__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_4, (int)fre[i]);
	  		HAL_Delay(1);
	  	}

	  	HAL_Delay(1000-1);


//		__HAL_TIM_SET_COMPARE(&htim8, TIM_CHANNEL_4, 200);
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
