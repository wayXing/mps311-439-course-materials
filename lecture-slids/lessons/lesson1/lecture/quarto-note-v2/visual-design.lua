local figures = "figures/"
local function txt(b) return pandoc.utils.stringify(b) end
local function fig(path, alt, caption)
  local image = pandoc.Image(alt, path)
  return pandoc.Div({pandoc.Para({image}), pandoc.Para({pandoc.Emph(caption)})}, pandoc.Attr("", {"figure"}, {}))
end
local function wide(path, alt, caption, class)
  return pandoc.Div({fig(path, alt, caption)}, pandoc.Attr("", {class or "wide-figure"}, {}))
end

function Pandoc(doc)
  local out, section, intro_count = {}, "", 0
  local extension, takeaways, ai_lists = {}, {}, {}
  local in_extension, in_takeaways = false, false
  local function emit(b)
    if in_extension then table.insert(extension, b)
    elseif in_takeaways then table.insert(takeaways, b)
    else table.insert(out, b) end
  end
  local function flush_extension()
    if #extension > 0 then table.insert(out, pandoc.Div(extension, pandoc.Attr("", {"extension"}, {}))) end
    extension = {}
  end

  for _, b in ipairs(doc.blocks) do
    if b.t == "Header" then
      local h = txt(b)
      if b.level ~= 1 then
        if in_extension and b.level == 2 and not h:match("Optional MPS439") then flush_extension(); in_extension = false end
        section = h
        if h:match("Optional MPS439") then in_extension = true end
        if h == "9. Takeaways" then in_takeaways = true; table.insert(out, b)
        else
          if b.level == 3 and (h == "Predict a number" or h == "Support a classification decision" or h == "Find structure without supplied answers" or h == "Learn useful representations") then b.classes:insert("course-stage")
          elseif b.level == 3 then b.classes:insert("reasoning-question") end
          emit(b)
        end
      end
    elseif b.t == "RawBlock" and b.format == "html" and b.text:match("Figure task") then
      if b.text:match("actual market level and the AI%-generated") then
        emit(wide(figures.."fig_ai_prediction.png", "Actual market and AI prediction", "At this scale, the AI-generated prediction appears almost indistinguishable from the actual market."))
      elseif b.text:match("same period and visual scale") then
        local pair = pandoc.Div({
          fig(figures.."fig_ai_prediction.png", "Actual market and AI prediction", "A model-generated prediction looks remarkably close."),
          fig(figures.."fig_yesterday_baseline.png", "Actual market and yesterday's closing level", "A rule using yesterday's value looks equally persuasive."),
          pandoc.Div({pandoc.Para({pandoc.Strong("A convincing-looking prediction is not yet evidence of useful knowledge.")})}, pandoc.Attr("", {"pair-conclusion"}, {}))
        }, pandoc.Attr("", {"evidence-pair"}, {})); emit(pair)
      elseif b.text:match("recurring modelling process") then emit(wide(figures.."fig_modelling_path.png", "The six connected modelling questions", "A reusable path from a real question to a conclusion that the evidence can support."))
      elseif b.text:match("four course stages") then emit(wide(figures.."fig_course_progression.png", "Four cumulative stages of modelling capability", "Each stage expands the range of problems that students can handle.")) end
    elseif b.t == "BlockQuote" then
      local t = txt(b)
      if t:match("Use historical stock") then emit(pandoc.Div({b}, pandoc.Attr("", {"note-prompt"}, {})))
      elseif t:match("Were we trying") then emit(pandoc.Div({b}, pandoc.Attr("", {"note-question"}, {})))
      elseif t:match("Predict that today's") or t:match("Use, test, check") then emit(pandoc.Div({b}, pandoc.Attr("", {"key-insight"}, {})))
      elseif t:match("Using only information") then emit(pandoc.Div({b}, pandoc.Attr("", {"note-prompt", "revised-prompt"}, {})))
      else emit(b) end
    elseif b.t == "BulletList" and section == "What you should gain from this lesson" then emit(pandoc.Div({b}, pandoc.Attr("", {"learning-outcomes"}, {})))
    elseif b.t == "OrderedList" and section == "4. What you should be able to do by the end of the course" then emit(pandoc.Div({b}, pandoc.Attr("", {"capability-list"}, {})))
    elseif b.t == "BulletList" and section == "6. How we will learn each week" then emit(pandoc.Div({b}, pandoc.Attr("", {"role-map"}, {})))
    elseif b.t == "BulletList" and section == "7. Learning with AI" then
      table.insert(ai_lists, b)
      if #ai_lists == 2 then emit(pandoc.Div({
        pandoc.Div({pandoc.Header(3, "AI can accelerate"), ai_lists[1]}, pandoc.Attr("", {"ai-can-help"}, {})),
        pandoc.Div({pandoc.Header(3, "You remain responsible"), ai_lists[2]}, pandoc.Attr("", {"you-own"}, {}))
      }, pandoc.Attr("", {"responsibility-grid"}, {}))) end
    elseif b.t == "Para" then
      local t = txt(b)
      local intro = false
      if section == "" and intro_count < 3 then intro_count = intro_count + 1; intro = true end
      if t:match("Market levels usually change") then
        emit(b)
      elseif t:match("Look more closely at a short") then
        emit(b); emit(wide(figures.."fig_level_vs_change.png", "A close-up of daily changes and a full-period baseline comparison", "The last 20 days make individual misses visible; across all unseen days, the model gets about 48 directions right per 100, while an always-up rule gets about 54.", "bridge-figure"))
      elseif section == "Applying the six questions to the stock example" and t:match("^What") then emit(pandoc.Div({b}, pandoc.Attr("", {"audit-step"}, {})))
      elseif t:match("This audit used no new algorithm") or t:match("The course therefore does not move") then emit(pandoc.Div({b}, pandoc.Attr("", {"key-insight"}, {})))
      elseif t:match("Before finishing, check whether") then emit(pandoc.Div({b}, pandoc.Attr("", {"ability-check"}, {})))
      elseif t:match("Return to the opening stock%-market graph") then emit(pandoc.Div({b}, pandoc.Attr("", {"note-question"}, {})))
      elseif section == "7. Learning with AI" and (t:match("AI assistants are useful") or t == "You remain responsible for:") then
      elseif intro then emit(pandoc.Div({b}, pandoc.Attr("", {"lead-intro"}, {})))
      else emit(b) end
    else emit(b) end
  end
  if in_extension then flush_extension() end
  if in_takeaways and #takeaways > 0 then table.insert(out, pandoc.Div(takeaways, pandoc.Attr("", {"takeaways"}, {}))) end
  doc.blocks = out
  return doc
end
