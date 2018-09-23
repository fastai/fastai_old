# -*- encoding: utf-8 -*-
# stub: ruby-enum 0.7.2 ruby lib

Gem::Specification.new do |s|
  s.name = "ruby-enum"
  s.version = "0.7.2"

  s.required_rubygems_version = Gem::Requirement.new(">= 1.3.6") if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib"]
  s.authors = ["Daniel Doubrovkine"]
  s.date = "2018-02-16"
  s.email = "dblock@dblock.org"
  s.homepage = "http://github.com/dblock/ruby-enum"
  s.licenses = ["MIT"]
  s.rubygems_version = "2.5.2.1"
  s.summary = "Enum-like behavior for Ruby."

  s.installed_by_version = "2.5.2.1" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<i18n>, [">= 0"])
    else
      s.add_dependency(%q<i18n>, [">= 0"])
    end
  else
    s.add_dependency(%q<i18n>, [">= 0"])
  end
end
