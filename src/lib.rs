struct Solution;

#[allow(dead_code)]
impl Solution {
    // 76. Minimum Window Substring
    pub fn min_window(s: String, t: String) -> String {
        let s_bytes = s.as_bytes();
        let t_bytes = t.as_bytes();
        let mut map = std::collections::HashMap::new();
        for c in t_bytes.iter() {
            *map.entry(c).or_insert(0) += 1;
        }
        let mut left = 0;
        let mut count = 0;
        let mut min_len = std::usize::MAX;
        let mut min_left = 0;
        for (right, c) in s_bytes.iter().enumerate() {
            if let Some(&v) = map.get(&c) {
                if v > 0 {
                    count += 1;
                }
                map.insert(c, v - 1);
            }
            while count == t.len() {
                if right - left + 1 < min_len {
                    min_len = right - left + 1;
                    min_left = left;
                }
                if let Some(&v) = map.get(&s_bytes[left]) {
                    if v == 0 {
                        count -= 1;
                    }
                    map.insert(&s_bytes[left], v + 1);
                }
                left += 1;
            }
        }
        if min_len == std::usize::MAX {
            "".to_string()
        } else {
            s[min_left..min_left + min_len].to_string()
        }
    }

    // 30. Substring with Concatenation of All Words
    pub fn find_substring(s: String, words: Vec<String>) -> Vec<i32> {
        if s.is_empty() || words.is_empty() {
            return vec![];
        }
        let mut res = vec![];
        let words_len = words.len();
        let mut map = std::collections::HashMap::with_capacity(words_len);
        let word_len = words[0].len();
        let s_len = s.len();
        for word in words.iter() {
            *map.entry(word.as_str()).or_insert(0) += 1;
        }
        for i in 0..word_len {
            let mut left = i;
            let mut right = i;
            let mut count = 0;
            let mut window = std::collections::HashMap::with_capacity(words_len);
            while right + word_len <= s_len {
                let word = &s[right..right + word_len];
                right += word_len;
                if !map.contains_key(word) {
                    count = 0;
                    left = right;
                    window.clear();
                } else {
                    *window.entry(word).or_insert(0) += 1;
                    count += 1;
                    while window.get(word).unwrap() > map.get(word).unwrap() {
                        let left_word = &s[left..left + word_len];
                        left += word_len;
                        *window.get_mut(left_word).unwrap() -= 1;
                        count -= 1;
                    }
                    if count == words_len {
                        res.push(left as i32);
                    }
                }
            }
        }
        res
    }

    // 3. Longest Substring Without Repeating Characters
    pub fn length_of_longest_substring(s: String) -> i32 {
        let mut max = 0;
        let mut left = 0;
        let mut map = std::collections::HashMap::new();
        for (right, c) in s.chars().enumerate() {
            if let Some(&index) = map.get(&c) {
                left = left.max(index + 1);
            }
            map.insert(c, right);
            max = max.max(right - left + 1);
        }
        max as i32
    }

    // 209. Minimum Size Subarray Sum
    pub fn min_sub_array_len(target: i32, nums: Vec<i32>) -> i32 {
        let mut min_len = std::i32::MAX;
        let mut sum = 0;
        let mut left = 0;
        for (right, &num) in nums.iter().enumerate() {
            sum += num;
            while sum >= target {
                min_len = min_len.min((right - left + 1) as i32);
                sum -= nums[left];
                left += 1;
            }
        }
        if min_len == std::i32::MAX {
            0
        } else {
            min_len
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn mini_window() {
        match fs::read_to_string("mini_window.txt") {
            Ok(content) => {
                let mut lines = content.lines();
                if let (Some(s), Some(t)) = (lines.next(), lines.next()) {
                    let res = Solution::min_window(s.to_string(), t.to_string());
                    assert_eq!(res, "");
                }
            }
            Err(_) => panic!("File not found"),
        }
    }
}
