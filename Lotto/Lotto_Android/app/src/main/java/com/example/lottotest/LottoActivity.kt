package com.example.lottotest

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.lottotest.databinding.ActivityLottoBinding
import retrofit2.Call
import retrofit2.Response

class LottoActivity : AppCompatActivity() {
    private lateinit var binding: ActivityLottoBinding
    private val items = mutableListOf<Item>()
    private lateinit var adapter: ItemAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityLottoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.backbutton.setOnClickListener {
            finish()
        }

        val num = 1120
        binding.lottoTextView.text = "예상된 $num 회차 당첨 번호"

        adapter = ItemAdapter(items)  // 어댑터 초기화
        binding.recyclerView.layoutManager = LinearLayoutManager(this)
        binding.recyclerView.adapter = adapter  // RecyclerView에 어댑터 설정

        binding.predictButton.setOnClickListener {
            Log.d("ny", "button click")
            ApiRetrofit.apiService.getData().enqueue(object : retrofit2.Callback<ApiDataResponse> {
                override fun onResponse(call: Call<ApiDataResponse>, response: Response<ApiDataResponse>) {
                    fun formatPrediction(prediction: String): String {
                        // 숫자만 추출하기 위한 정규 표현식
                        val regex = "\\d+".toRegex()
                        // 모든 숫자를 찾아 리스트로 변환
                        val numbers = regex.findAll(prediction).map { it.value }.toList()
                        // 숫자들을 콤마로 구분된 문자열로 변환
                        return numbers.joinToString(", ")
                    }
                    if (response.isSuccessful) {
                        response.body()?.let {
                            items.clear()  // 기존 데이터를 지우고 새로운 데이터 추가
                            items.addAll(listOf(
                                Item(1, "CNN", formatPrediction(it.cnn_prediction)),
                                Item(2, "MLP", formatPrediction(it.mlp_prediction)),
                                Item(3, "Random Forest", formatPrediction(it.rf_prediction)),
                                Item(4, "RNN", formatPrediction(it.rnn_prediction)),
                                Item(5, "Transformer", formatPrediction(it.trans_prediction))
                            ))
                            adapter.notifyDataSetChanged()  // 데이터 변경 알림
                        } ?: Log.d("ny", "Received null response body")
                    } else {
                        Log.d("ny", "Response not successful: ${response.code()}")
                    }
                }

                override fun onFailure(call: Call<ApiDataResponse>, t: Throwable) {
                    Log.e("API Error", "API load failure: ${t.message}")
                }
            })
        }

        binding.randomButton.setOnClickListener {
            val text = generateRandomNumbersAsString()
            // "랜덤추출" 아이템 찾기
            val randomItemIndex = items.indexOfFirst { it.modelName == "랜덤추출" }

            if (randomItemIndex != -1) {
                // 아이템이 존재하면, 텍스트만 업데이트
                items[randomItemIndex].lottoNum = text
            } else {
                // 아이템이 없으면, 새로운 아이템 추가
                items.add(Item(6, "랜덤추출", text))
            }
            adapter.notifyDataSetChanged()  // 데이터 변경 알림
        }
    }

    private fun generateRandomNumbersAsString(): String {
        val numbers = (1..45).shuffled().take(6).sorted()
        return numbers.joinToString(", ")
    }
}
