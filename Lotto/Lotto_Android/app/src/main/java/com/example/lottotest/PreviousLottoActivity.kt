package com.example.lottotest

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.example.lottotest.databinding.ActivityPreviousLottoBinding
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class PreviousLottoActivity : AppCompatActivity() {
    private lateinit var binding: ActivityPreviousLottoBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_previous_lotto)

        binding = ActivityPreviousLottoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.backbutton.setOnClickListener {
            finish()
        }

        fetchData()
    }

    private fun fetchData() {
        val call = ApiRetrofit.apiService.getLotteryResult()
        call.enqueue(object : Callback<LotteryResult> {
            override fun onResponse(call: Call<LotteryResult>, response: Response<LotteryResult>) {
                if (response.isSuccessful) {
                    val dataResponse = response.body()
                    if (dataResponse != null) {
                        binding.drawNo.text = dataResponse.drawNo.toString() + " 회 당첨결과"
                        binding.drawNum1.text = dataResponse.numbers[0].toString()
                        binding.drawNum2.text = dataResponse.numbers[1].toString()
                        binding.drawNum3.text = dataResponse.numbers[2].toString()
                        binding.drawNum4.text = dataResponse.numbers[3].toString()
                        binding.drawNum5.text = dataResponse.numbers[4].toString()
                        binding.drawNum6.text = dataResponse.numbers[5].toString()
                        binding.drawBonusNum.text = dataResponse.bonusNumber.toString()
                    }

                } else {
                    // 서버 요청이 실패한 경우
                }
            }

            override fun onFailure(call: Call<LotteryResult>, t: Throwable) {
                TODO("Not yet implemented")
            }
        })
    }
}